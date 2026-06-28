"""Main module for TARVI (Transcription-Factor Aided RNA Velocity Inference)."""
from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl

from ._constants import REGISTRY_KEYS
from ._gnn import VelocityRefiner
from ._tf_module import TFRegulatedTranscription


class _DirichletKLFunction(torch.autograd.Function):
    """Custom autograd function that computes Dirichlet KL on CPU (avoids CUDA lgamma bugs)."""

    @staticmethod
    def forward(ctx, alpha_p, alpha_q):
        # Compute on CPU to avoid CUDA lgamma driver error
        p_cpu = alpha_p.detach().cpu().float()
        q_cpu = alpha_q.detach().cpu().float()

        sum_p = p_cpu.sum(dim=-1)
        sum_q = q_cpu.sum(dim=-1)

        t1 = torch.lgamma(sum_p) - torch.lgamma(sum_q)
        t2 = (torch.lgamma(q_cpu) - torch.lgamma(p_cpu)).sum(dim=-1)
        t3 = ((p_cpu - q_cpu) * (torch.digamma(p_cpu) - torch.digamma(sum_p).unsqueeze(-1))).sum(dim=-1)

        result = (t1 + t2 + t3).to(alpha_p.device)

        # Save for backward (compute grad of KL w.r.t. alpha_p on CPU)
        ctx.save_for_backward(alpha_p, alpha_q)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        alpha_p, alpha_q = ctx.saved_tensors
        p_cpu = alpha_p.detach().cpu().float()

        sum_p = p_cpu.sum(dim=-1, keepdim=True)
        # d KL / d alpha_p_k = trigamma(alpha_p_k) * (alpha_p_k - alpha_q_k)
        #                     - trigamma(sum_p) * sum_k(alpha_p_k - alpha_q_k)
        #                     + digamma(alpha_p_k) - digamma(sum_p)
        # Simplified: d/d(alpha_p_k) = digamma(sum_p) - digamma(alpha_p_k)
        #             + digamma(alpha_p_k) - digamma(sum_p)
        #             + ... (no, let me just differentiate directly)
        # KL = lgamma(sum_p) - sum lgamma(p_k) + sum (p_k-q_k)(psi(p_k) - psi(sum_p))
        # d/dp_k = psi(sum_p) - psi(p_k) + (psi(p_k) - psi(sum_p))
        #        + (p_k - q_k)(psi'(p_k) - psi'(sum_p))
        # = (p_k - q_k)(trigamma(p_k) - trigamma(sum_p))
        q_cpu = alpha_q.detach().cpu().float()
        trigamma_p = torch.special.polygamma(1, p_cpu)
        trigamma_sum_p = torch.special.polygamma(1, sum_p)

        grad_p = (p_cpu - q_cpu) * (trigamma_p - trigamma_sum_p)
        grad_p = grad_p.to(alpha_p.device)

        return grad_output.unsqueeze(-1) * grad_p, None


def _kl_dirichlet_safe(alpha_p: torch.Tensor, alpha_q: torch.Tensor) -> torch.Tensor:
    """Compute KL(Dirichlet(alpha_p) || Dirichlet(alpha_q)) with gradients, CPU-safe lgamma."""
    # Clamp to safe range
    alpha_p = torch.clamp(alpha_p, min=0.01, max=1e6)
    alpha_q = torch.clamp(alpha_q, min=0.01, max=1e6)
    return _DirichletKLFunction.apply(alpha_p, alpha_q)

torch.backends.cudnn.benchmark = True


class DecoderTARVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    linear_decoder
        Whether to use linear decoder for time
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        linear_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_ouput = n_output
        self.linear_decoder = linear_decoder
        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        self.pi_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # categorical pi - 4 states
        self.px_pi_decoder = nn.Linear(n_hidden, 4 * n_output)

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        # tau for repression
        self.px_tau_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        self.linear_scaling_tau = nn.Parameter(torch.zeros(n_output))
        self.linear_scaling_tau_intercept = nn.Parameter(torch.zeros(n_output))

    def forward(self, z: torch.Tensor, latent_dim: int = None):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the distribution of expression

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        latent_dim
            optional latent dimension to mask

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            parameters for the distribution of expression

        """
        z_in = z
        if latent_dim is not None:
            mask = torch.zeros_like(z)
            mask[..., latent_dim] = 1
            z_in = z * mask
        # The decoder returns values for the parameters of the distribution
        rho_first = self.rho_first_decoder(z_in)

        if not self.linear_decoder:
            px_rho = self.px_rho_decoder(rho_first)
            px_tau = self.px_tau_decoder(rho_first)
        else:
            px_rho = nn.Sigmoid()(rho_first)
            px_tau = 1 - nn.Sigmoid()(
                rho_first * self.linear_scaling_tau.exp()
                + self.linear_scaling_tau_intercept
            )

        # cells by genes by 4
        pi_first = self.pi_first_decoder(z)
        px_pi = nn.Softplus()(
            torch.reshape(self.px_pi_decoder(pi_first), (z.shape[0], self.n_ouput, 4))
        )

        return px_pi, px_rho, px_tau


# VAE model
class TARVIVAE(BaseModuleClass):
    """Variational auto-encoder model for TARVI.

    This extends VeloVI with TF-regulated transcription (ENCODE/ChEA databases),
    ODE residual supervision, and velocity confidence weighting.

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of 'normal' or 'ln'
    use_layer_norm
        Whether to use layer norm in layers
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
    use_tf_regulation
        Whether to use TF-regulated transcription rates.
    tf_target_mask
        Binary mask [n_genes, n_tfs] for TF regulation.
    n_tfs
        Number of transcription factors.
    tf_indices
        Indices of TF genes in the gene list.
    tf_init_weights
        Initial weights for TF-target pairs.
    ode_residual_weight
        Weight for ODE residual time supervision loss (best=1.0 from ablation).
    velo_confidence_weight
        Weight for velocity confidence loss (best=0.0 with VPT post-processing).
    """

    def __init__(
        self,
        n_input: int,
        true_time_switch: Optional[np.ndarray] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        gamma_unconstr_init: Optional[np.ndarray] = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        alpha_1_unconstr_init: Optional[np.ndarray] = None,
        lambda_alpha_unconstr_init: Optional[np.ndarray] = None,
        switch_spliced: Optional[np.ndarray] = None,
        switch_unspliced: Optional[np.ndarray] = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
        dirichlet_concentration: float = 0.25,
        linear_decoder: bool = False,
        time_dep_transcription_rate: bool = False,
        # TARVI-specific parameters
        use_tf_regulation: bool = False,
        tf_target_mask: Optional[np.ndarray] = None,
        n_tfs: int = 0,
        tf_indices: Optional[np.ndarray] = None,
        tf_init_weights: Optional[np.ndarray] = None,
        tf_l1_weight: float = 0.01,
        # improvements
        velocity_consistency_weight: float = 0.2,
        temporal_smoothness_weight: float = 0.1,
        # learnable latent time
        use_learned_time: bool = True,
        time_reg_weight: float = 1.0,
        # GNN velocity refinement
        use_gnn_refinement: bool = False,
        gnn_hidden_dim: int = 128,
        gnn_n_heads: int = 4,
        gnn_n_layers: int = 2,
        gnn_dropout: float = 0.1,
        gnn_coherence_weight: float = 0.5,
        # scVelo-inspired improvements (best values from ablation study)
        ode_residual_weight: float = 1.0,     # direct ODE residual time supervision (best=1.0)
        velo_confidence_weight: float = 0.0,  # mean-centered neighbor cosine sim (best=0.0 with VPT)
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.time_dep_transcription_rate = time_dep_transcription_rate
        self.use_tf_regulation = use_tf_regulation
        self.tf_l1_weight = tf_l1_weight
        self._current_epoch_frac = 0.0  # updated by training hook
        # velocity-time consistency and temporal smoothness
        self.velocity_consistency_weight = velocity_consistency_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
        # learnable latent time head
        self.use_learned_time = use_learned_time
        self.time_reg_weight = time_reg_weight
        # GNN velocity refinement
        self.use_gnn_refinement = use_gnn_refinement
        self.gnn_coherence_weight = gnn_coherence_weight
        # scVelo-inspired
        self.ode_residual_weight = ode_residual_weight
        self.velo_confidence_weight = velo_confidence_weight

        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        n_genes = n_input * 2

        # switching time
        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_input))
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None

        # degradation
        if gamma_unconstr_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(gamma_unconstr_init)
            )

        # splicing
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_input))

        # transcription
        if alpha_unconstr_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_unconstr_init)
            )

        if alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_1_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_1_unconstr_init)
            )
        self.alpha_1_unconstr.requires_grad = time_dep_transcription_rate

        if lambda_alpha_unconstr_init is None:
            self.lambda_alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.lambda_alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(lambda_alpha_unconstr_init)
            )
        self.lambda_alpha_unconstr.requires_grad = time_dep_transcription_rate

        # likelihood dispersion (Gaussian scale)
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_genes, 4))

        # === TF Regulation Module (Phase 2) ===
        if use_tf_regulation and tf_target_mask is not None:
            # When TF regulation is active, alpha_unconstr serves as fallback/init
            # but the TF module produces cell-specific alpha
            self.alpha_unconstr.requires_grad = False  # disable static alpha learning

            basal_init = self.alpha_unconstr.data.clone()
            mask_tensor = torch.from_numpy(tf_target_mask)

            init_w = None
            if tf_init_weights is not None:
                init_w = torch.from_numpy(tf_init_weights)

            self.tf_module = TFRegulatedTranscription(
                n_genes=n_input,
                n_tfs=n_tfs,
                tf_target_mask=mask_tensor,
                init_weights=init_w,
                basal_alpha_init=basal_init,
            )
            self.register_buffer("tf_indices", torch.from_numpy(tf_indices).long())
        else:
            self.tf_module = None
            self.tf_indices = None

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderTARVI(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
        )

        # Deeper latent time head with residual: z -> scalar time in [0, 1]
        if use_learned_time:
            self.time_head_fc1 = nn.Linear(n_latent, n_hidden)
            self.time_head_fc2 = nn.Linear(n_hidden, n_hidden)
            self.time_head_out = nn.Linear(n_hidden, 1)
            self.time_head_skip = nn.Linear(n_latent, n_hidden)
            self.time_head_ln = nn.LayerNorm(n_hidden)

        # GNN velocity refiner — smooths velocity using cell-cell graph
        if use_gnn_refinement:
            self.velocity_refiner = VelocityRefiner(
                n_genes=n_input,
                z_dim=n_latent,
                hidden_dim=gnn_hidden_dim,
                n_heads=gnn_n_heads,
                n_layers=gnn_n_layers,
                dropout=gnn_dropout,
            )
        else:
            self.velocity_refiner = None
        # Edge index buffer — set externally via set_edge_index()
        self._edge_index = None

    def set_edge_index(self, edge_index: torch.Tensor):
        """Set the k-NN graph edge index for GNN velocity refinement.

        Parameters
        ----------
        edge_index
            [2, E] tensor of source-destination edges from k-NN graph.
        """
        self._edge_index = edge_index

    def _get_inference_input(self, tensors):
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        input_dict = {
            "spliced": spliced,
            "unspliced": unspliced,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]
        alpha_1 = inference_outputs["alpha_1"]
        lambda_alpha = inference_outputs["lambda_alpha"]

        input_dict = {
            "z": z,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha,
            "alpha_1": alpha_1,
            "lambda_alpha": lambda_alpha,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        spliced,
        unspliced,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        # Get kinetic rates (alpha may be cell-specific with TF regulation)
        if self.use_tf_regulation and self.tf_module is not None:
            tf_expression = spliced[:, self.tf_indices]
            gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates(
                tf_expression=tf_expression
            )
        else:
            gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        # Learned latent time from z (deeper with residual)
        learned_time = None
        if self.use_learned_time:
            h = F.relu(self.time_head_fc1(z))
            h = h + self.time_head_skip(z)  # residual from input
            h = self.time_head_ln(h)
            h = F.relu(self.time_head_fc2(h))
            learned_time = torch.sigmoid(self.time_head_out(h)).squeeze(-1)  # [batch] in [0, 1]

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "qzm": qz_m,
            "qzv": qz_v,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha,
            "alpha_1": alpha_1,
            "lambda_alpha": lambda_alpha,
            "learned_time": learned_time,
        }
        return outputs

    def _get_rates(self, tf_expression: Optional[torch.Tensor] = None):
        # globals
        # degradation
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        # splicing
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)

        # transcription
        if self.use_tf_regulation and self.tf_module is not None and tf_expression is not None:
            # Cell-specific alpha from TF module: [batch, n_genes]
            alpha = self.tf_module(tf_expression)
        else:
            # Static per-gene alpha (original VeloVI behavior)
            alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 50)

        if self.time_dep_transcription_rate:
            alpha_1 = torch.clamp(F.softplus(self.alpha_1_unconstr), 0, 50)
            lambda_alpha = torch.clamp(F.softplus(self.lambda_alpha_unconstr), 0, 50)
        else:
            alpha_1 = self.alpha_1_unconstr
            lambda_alpha = self.lambda_alpha_unconstr

        return gamma, beta, alpha, alpha_1, lambda_alpha

    @auto_move_data
    def generative(self, z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha, px_rho, px_tau = self.decoder(decoder_input, latent_dim=latent_dim)
        px_pi = Dirichlet(px_pi_alpha).rsample()

        scale_unconstr = self.scale_unconstr
        scale = F.softplus(scale_unconstr)

        # Gaussian mixture likelihood
        mixture_dist_s, mixture_dist_u, end_penalty = self.get_px(
            px_pi,
            px_rho,
            px_tau,
            scale,
            gamma,
            beta,
            alpha,
            alpha_1,
            lambda_alpha,
        )
        return {
            "px_pi": px_pi,
            "px_rho": px_rho,
            "px_tau": px_tau,
            "scale": scale,
            "px_pi_alpha": px_pi_alpha,
            "mixture_dist_u": mixture_dist_u,
            "mixture_dist_s": mixture_dist_s,
            "end_penalty": end_penalty,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        # Always use Gaussian mixture for reconstruction (preserves pseudotime)
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]
        reconst_loss_s = -mixture_dist_s.log_prob(spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)
        reconst_loss = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)

        # Use CPU-safe KL to avoid CUDA lgamma driver errors
        kl_pi = _kl_dirichlet_safe(
            px_pi_alpha,
            self.dirichlet_concentration * torch.ones_like(px_pi),
        ).sum(dim=-1)

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = torch.mean(reconst_loss + weighted_kl_local)

        loss = local_loss + self.penalty_scale * (1 - kl_weight) * end_penalty

        # TF weight L1 regularization (mild sparsity)
        if self.use_tf_regulation and self.tf_module is not None:
            tf_l1 = self.tf_l1_weight * self.tf_module.w_raw.abs().mean()
            loss = loss + tf_l1

        # === Learned Time Reconstruction Loss ===
        # Train the time head to match ODE mixture-weight time (indirect supervision)
        if self.use_learned_time and inference_outputs.get("learned_time") is not None:
            learned_time = inference_outputs["learned_time"]  # [batch] in [0, 1]
            ode_time = self._compute_ode_time(generative_outputs).detach()
            time_recon_loss = F.mse_loss(learned_time, ode_time)
            loss = loss + self.time_reg_weight * time_recon_loss

            # === ODE Residual Time Loss (scVelo-inspired direct EM supervision) ===
            # Directly checks |u_ode(t_i) - u_obs|² + |s_ode(t_i) - s_obs|²
            # This is the key insight from scVelo's EM that gives PT Spearman=0.89
            if self.ode_residual_weight > 0:
                # Directly check |u_ode(t_i) - u_obs|² + |s_ode(t_i) - s_obs|²
                # Gradient flows through learned_time → time head (scVelo EM-inspired)
                ode_res_loss = self._compute_ode_residual_loss(
                    spliced, unspliced, inference_outputs, learned_time
                )
                loss = loss + self.ode_residual_weight * ode_res_loss

        # === Velocity-Time Consistency Loss ===
        # Encourages velocity to point in direction of increasing latent time
        # Uses within-batch pairwise comparisons
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        if self.velocity_consistency_weight > 0:
            vtc_loss = self._velocity_time_consistency(
                spliced, unspliced, gamma, beta, generative_outputs,
                learned_time=inference_outputs.get("learned_time"),
            )
            loss = loss + self.velocity_consistency_weight * vtc_loss

        # === Temporal Smoothness Loss ===
        # Neighboring cells in latent space should have similar latent times
        if self.temporal_smoothness_weight > 0:
            ts_loss = self._temporal_smoothness(
                inference_outputs["qz_m"], generative_outputs["px_rho"],
                generative_outputs["px_tau"], generative_outputs["px_pi"],
                learned_time=inference_outputs.get("learned_time"),
            )
            loss = loss + self.temporal_smoothness_weight * ts_loss

        # === Velocity Confidence Loss (scVelo-inspired mean-centered cosine sim) ===
        # Maximizes alignment between each cell's velocity and its latent-space neighbors
        if self.velo_confidence_weight > 0:
            raw_velocity = self._compute_raw_velocity(tensors, inference_outputs, generative_outputs)
            vc_loss = self._compute_velo_confidence_loss(raw_velocity, inference_outputs["z"])
            loss = loss + self.velo_confidence_weight * vc_loss

        # === Velocity Coherence Loss ===
        # Directly penalize velocity inconsistency between latent-space neighbors
        # Only active when use_gnn_refinement=True (the full_gnn mode)
        if self.use_gnn_refinement and self.gnn_coherence_weight > 0:
            raw_velocity = self._compute_raw_velocity(tensors, inference_outputs, generative_outputs)
            z = inference_outputs["z"]
            batch_size = z.shape[0]

            if batch_size >= 16:
                with torch.no_grad():
                    k = min(10, batch_size - 1)
                    dists = torch.cdist(z.detach(), z.detach())
                    dists.fill_diagonal_(float('inf'))
                    _, nn_idx = dists.topk(k, largest=False)  # [batch, k]

                # Cosine similarity between cell's velocity and each neighbor's velocity
                nn_vels = raw_velocity[nn_idx]  # [batch, k, genes]
                cell_vel = raw_velocity.unsqueeze(1).expand_as(nn_vels)  # [batch, k, genes]
                cos_sim = F.cosine_similarity(cell_vel, nn_vels, dim=-1)  # [batch, k]
                coherence_loss = (1.0 - cos_sim).mean()

                loss = loss + self.gnn_coherence_weight * coherence_loss

        loss_recoder = LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local
        )

        return loss_recoder

    def _compute_ode_residual_loss(self, spliced, unspliced, inference_outputs, learned_time):
        """scVelo-inspired ODE residual loss for time supervision.

        Instead of regressing learned_time against mixture-weight ODE time,
        we directly evaluate |u_ode(t_i) - u_obs|² + |s_ode(t_i) - s_obs|²
        using the learned time t_i. This mirrors scVelo's EM time step.

        The ODE mean at time t under the dominant state (induction or repression)
        is computed and compared against observed expression.
        """
        alpha = inference_outputs["alpha"]
        alpha_1 = inference_outputs["alpha_1"]
        lambda_alpha = inference_outputs["lambda_alpha"]
        beta = inference_outputs["beta"]
        gamma = inference_outputs["gamma"]

        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        # Scale learned time [0,1] to actual time units [0, t_max]
        t_i = learned_time * self.t_max  # [batch]

        # Induction branch: use t_i clamped to [0, t_s] per gene
        t_s_exp = t_s.unsqueeze(0)  # [1, genes]
        t_i_exp = t_i.unsqueeze(-1).expand(-1, self.n_input)  # [batch, genes]
        t_ind = torch.min(t_i_exp, t_s_exp).clamp(min=0)  # [batch, genes]
        u_ind, s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_ind
        )

        # Repression branch: t_i >= t_s → use repression ODE
        u_0, s_0 = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s
        )
        t_rep_raw = (t_i_exp - t_s_exp).clamp(min=0)  # [batch, genes]
        u_rep, s_rep = self._get_repression_unspliced_spliced(
            u_0, s_0, beta, gamma, t_rep_raw
        )

        # Soft-select between branches using time position relative to switch
        # gate≈1 when t_i < t_s (induction), gate≈0 when t_i > t_s (repression)
        gate = torch.sigmoid(5.0 * (t_s_exp - t_i_exp))  # [batch, genes]
        u_pred = gate * u_ind + (1 - gate) * u_rep
        s_pred = gate * s_ind + (1 - gate) * s_rep

        # Normalize to [0,1] scale per gene to balance high/low expressed genes
        eps = 1e-6
        s_scale = spliced.abs().mean(dim=0, keepdim=True).clamp(min=eps)
        u_scale = unspliced.abs().mean(dim=0, keepdim=True).clamp(min=eps)

        res_s = ((s_pred - spliced) / s_scale).pow(2).mean()
        res_u = ((u_pred - unspliced) / u_scale).pow(2).mean()

        return 0.5 * (res_s + res_u)

    def _compute_velo_confidence_loss(self, velocity, z):
        """scVelo-inspired velocity confidence loss.

        Maximizes mean-centered cosine similarity between each cell's velocity
        and its k-NN neighbors in latent space. Mirrors scVelo's velocity_confidence.

        V_i_centered = V_i - mean(V_i)
        confidence_i = mean_j [ (V_j_centered · V_i_centered) / (|V_j| |V_i|) ]
        Loss = 1 - mean_i(confidence_i)
        """
        batch_size = velocity.shape[0]
        if batch_size < 16:
            return torch.tensor(0.0, device=velocity.device)

        k = min(10, batch_size - 1)
        with torch.no_grad():
            dists = torch.cdist(z.detach(), z.detach())
            dists.fill_diagonal_(float('inf'))
            _, nn_idx = dists.topk(k, largest=False)  # [batch, k]

        # Mean-center velocities (scVelo subtracts row mean before computing sim)
        V = velocity - velocity.mean(dim=1, keepdim=True)  # [batch, genes]
        V_norm = V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        V_unit = V / V_norm  # [batch, genes]

        # Neighbor velocities mean-centered individually
        V_nn = V[nn_idx]  # [batch, k, genes]
        V_nn_norm = V_nn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        V_nn_unit = V_nn / V_nn_norm  # [batch, k, genes]

        # Cosine similarity: V_i · V_j_neighbors
        cos_sim = (V_unit.unsqueeze(1) * V_nn_unit).sum(dim=-1)  # [batch, k]
        confidence = cos_sim.mean(dim=1)  # [batch]

        # Loss: push confidence toward 1
        loss = (1.0 - confidence.clamp(-1, 1)).mean()
        return loss

    def _compute_raw_velocity(self, tensors, inference_outputs, generative_outputs):
        """Compute raw ODE-derived velocity (ds/dt = beta*u - gamma*s).

        Uses the state-weighted expectation, same as get_velocity() in _model.py.
        Returns [batch, n_genes] velocity tensor.
        """
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        pi = generative_outputs["px_pi"]
        alpha = inference_outputs["alpha"]
        alpha_1 = inference_outputs["alpha_1"]
        lambda_alpha = inference_outputs["lambda_alpha"]
        beta = inference_outputs["beta"]
        gamma = inference_outputs["gamma"]
        rho = generative_outputs["px_rho"]
        tau = generative_outputs["px_tau"]

        switch_time = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        # Induction velocity
        ind_time = switch_time * rho
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, ind_time
        )
        velo_ind = beta * mean_u_ind - gamma * mean_s_ind

        # Repression velocity
        u_0, s_0 = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, switch_time
        )
        rep_time = (self.t_max - switch_time) * tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0, s_0, beta, gamma, rep_time
        )
        velo_rep = beta * mean_u_rep - gamma * mean_s_rep

        # Steady state velocity = 0
        velo_steady = torch.zeros_like(velo_ind)

        # State-weighted expectation
        ind_prob = pi[..., 0]
        steady_prob = pi[..., 1]
        rep_prob = pi[..., 2]

        velocity = ind_prob * velo_ind + rep_prob * velo_rep + steady_prob * velo_steady
        return velocity

    def _compute_ode_time(self, generative_outputs):
        """Compute per-cell time from ODE mixture weights, normalized to [0, 1]."""
        px_pi = generative_outputs["px_pi"]
        px_rho = generative_outputs["px_rho"]
        px_tau = generative_outputs["px_tau"]

        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        t_ind = t_s.unsqueeze(0) * px_rho
        t_ind_steady = t_s.unsqueeze(0).expand_as(t_ind)
        t_rep = t_s.unsqueeze(0) + (self.t_max - t_s.unsqueeze(0)) * px_tau
        t_rep_steady = torch.full_like(t_ind, self.t_max)

        t_states = torch.stack([t_ind, t_ind_steady, t_rep, t_rep_steady], dim=2)
        cell_time = (px_pi * t_states).sum(dim=2).mean(dim=1)  # [batch]

        # Normalize to [0, 1]
        t_min = cell_time.min()
        t_max = cell_time.max()
        cell_time_norm = (cell_time - t_min) / (t_max - t_min + 1e-8)
        return cell_time_norm

    def _velocity_time_consistency(self, spliced, unspliced, gamma, beta, generative_outputs, learned_time=None):
        """Velocity-time consistency: velocity should point toward increasing time.

        For random pairs (i, j) in the batch where t_i < t_j,
        the velocity of cell i projected onto (s_j - s_i) should be positive.
        """
        # Use learned time if available (smoother, better signal)
        if learned_time is not None:
            cell_time = learned_time  # [batch] in [0, 1]
        else:
            px_pi = generative_outputs["px_pi"]
            px_rho = generative_outputs["px_rho"]
            px_tau = generative_outputs["px_tau"]

            t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)
            t_ind = t_s.unsqueeze(0) * px_rho
            t_ind_steady = t_s.unsqueeze(0).expand_as(t_ind)
            t_rep = t_s.unsqueeze(0) + (self.t_max - t_s.unsqueeze(0)) * px_tau
            t_rep_steady = torch.full_like(t_ind, self.t_max)

            t_states = torch.stack([t_ind, t_ind_steady, t_rep, t_rep_steady], dim=2)
            cell_time = (px_pi * t_states).sum(dim=2).mean(dim=1)

        # Compute velocity: ds/dt = beta * u - gamma * s
        velocity_s = beta * unspliced - gamma * spliced  # [batch, genes]

        # Sample random pairs within batch
        batch_size = spliced.shape[0]
        n_pairs = min(batch_size // 2, 128)
        if n_pairs < 4:
            return torch.tensor(0.0, device=spliced.device)

        idx = torch.randperm(batch_size, device=spliced.device)[:2 * n_pairs]
        idx_i, idx_j = idx[:n_pairs], idx[n_pairs:]

        t_i, t_j = cell_time[idx_i], cell_time[idx_j]
        # Ensure t_i < t_j by swapping if needed
        swap = t_i > t_j
        t_i_final = torch.where(swap, t_j, t_i)
        t_j_final = torch.where(swap, t_i, t_j)
        idx_i_final = torch.where(swap, idx_j, idx_i)

        # Direction from i to j in expression space
        ds = spliced[idx_j] - spliced[idx_i_final]  # [n_pairs, genes]
        v_i = velocity_s[idx_i_final]  # [n_pairs, genes]

        # Cosine similarity between velocity and displacement
        cos_sim = F.cosine_similarity(v_i, ds, dim=1)  # [n_pairs]

        # Weight by time difference (stronger signal for larger time gaps)
        time_diff = (t_j_final - t_i_final).detach()
        weights = torch.clamp(time_diff / (time_diff.mean() + 1e-8), 0, 3)

        # Loss: penalize negative cosine similarity (velocity pointing wrong way)
        # Use soft margin: max(0, margin - cos_sim)
        margin = 0.0
        vtc_loss = (weights * F.relu(margin - cos_sim)).mean()

        return vtc_loss

    def _temporal_smoothness(self, qz_m, px_rho, px_tau, px_pi, learned_time=None):
        """Temporal smoothness: nearby cells in latent space should have similar times.

        Computes within-batch k-nearest neighbor time variance.
        """
        batch_size = qz_m.shape[0]
        if batch_size < 16:
            return torch.tensor(0.0, device=qz_m.device)

        # Use learned time if available
        if learned_time is not None:
            cell_time = learned_time
        else:
            t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)
            t_ind = t_s.unsqueeze(0) * px_rho
            t_ind_steady = t_s.unsqueeze(0).expand_as(t_ind)
            t_rep = t_s.unsqueeze(0) + (self.t_max - t_s.unsqueeze(0)) * px_tau
            t_rep_steady = torch.full_like(t_ind, self.t_max)

            t_states = torch.stack([t_ind, t_ind_steady, t_rep, t_rep_steady], dim=2)
            cell_time = (px_pi * t_states).sum(dim=2).mean(dim=1)

        # Find k nearest neighbors in latent space
        k = min(10, batch_size - 1)
        with torch.no_grad():
            dists = torch.cdist(qz_m, qz_m)  # [batch, batch]
            dists.fill_diagonal_(float('inf'))
            _, nn_idx = dists.topk(k, largest=False)  # [batch, k]

        # Time difference between cell and its neighbors
        nn_times = cell_time[nn_idx]  # [batch, k]
        time_diff = (cell_time.unsqueeze(1) - nn_times).pow(2)  # [batch, k]

        # Mean squared time difference with neighbors
        ts_loss = time_diff.mean()

        return ts_loss

    def get_px(
        self,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        beta,
        alpha,
        alpha_1,
        lambda_alpha,
    ) -> torch.Tensor:
        """Gaussian mixture likelihood."""
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # induction
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s * px_rho
        )

        if self.time_dep_transcription_rate:
            mean_u_ind_steady = (alpha_1 / beta).expand(n_cells, self.n_input)
            mean_s_ind_steady = (alpha_1 / gamma).expand(n_cells, self.n_input)
        else:
            # Handle cell-specific alpha (TF regulation) vs static alpha
            alpha_over_beta = alpha / beta
            alpha_over_gamma = alpha / gamma
            if alpha_over_beta.dim() == 1:
                mean_u_ind_steady = alpha_over_beta.expand(n_cells, self.n_input)
                mean_s_ind_steady = alpha_over_gamma.expand(n_cells, self.n_input)
            else:
                # Already [batch, n_genes] from TF module
                mean_u_ind_steady = alpha_over_beta
                mean_s_ind_steady = alpha_over_gamma

        # repression
        u_0, s_0 = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s
        )

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = torch.zeros_like(mean_u_ind)
        mean_s_rep_steady = torch.zeros_like(mean_u_ind)

        end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
            (s_0 - self.switch_spliced).pow(2)
        ).sum()

        # Stack means: [batch, genes, 4]
        mean_u = torch.stack(
            (mean_u_ind, mean_u_ind_steady, mean_u_rep, mean_u_rep_steady),
            dim=2,
        )
        mean_s = torch.stack(
            (mean_s_ind, mean_s_ind_steady, mean_s_rep, mean_s_rep_steady),
            dim=2,
        )

        # Learned per-gene scale (VeloVI baseline)
        scale_u_raw = scale[: self.n_input, :].expand(n_cells, self.n_input, 4).sqrt()
        scale_s_raw = scale[self.n_input :, :].expand(n_cells, self.n_input, 4).sqrt()

        learned_scale_u = torch.stack(
            (scale_u_raw[..., 0], scale_u_raw[..., 0],
             scale_u_raw[..., 0], 0.1 * scale_u_raw[..., 0]),
            dim=2,
        )
        learned_scale_s = torch.stack(
            (scale_s_raw[..., 0], scale_s_raw[..., 0],
             scale_s_raw[..., 0], 0.1 * scale_s_raw[..., 0]),
            dim=2,
        )

        scale_u = learned_scale_u
        scale_s = learned_scale_s

        dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)

        dist_s = Normal(mean_s, scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        return mixture_dist_s, mixture_dist_u, end_penalty

    def _get_induction_unspliced_spliced(
        self, alpha, alpha_1, lambda_alpha, beta, gamma, t, eps=1e-6
    ):
        if self.time_dep_transcription_rate:
            unspliced = alpha_1 / beta * (1 - torch.exp(-beta * t)) - (
                alpha_1 - alpha
            ) / (beta - lambda_alpha) * (
                torch.exp(-lambda_alpha * t) - torch.exp(-beta * t)
            )

            spliced = (
                alpha_1 / gamma * (1 - torch.exp(-gamma * t))
                + alpha_1
                / (gamma - beta + eps)
                * (torch.exp(-gamma * t) - torch.exp(-beta * t))
                - beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - lambda_alpha + eps)
                * (torch.exp(-lambda_alpha * t) - torch.exp(-gamma * t))
                + beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - beta + eps)
                * (torch.exp(-beta * t) - torch.exp(-gamma * t))
            )
        else:
            unspliced = (alpha / beta) * (1 - torch.exp(-beta * t))
            spliced = (alpha / gamma) * (1 - torch.exp(-gamma * t)) + (
                alpha / ((gamma - beta) + eps)
            ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))

        return unspliced, spliced

    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t, eps=1e-6):
        unspliced = torch.exp(-beta * t) * u_0
        spliced = s_0 * torch.exp(-gamma * t) - (
            beta * u_0 / ((gamma - beta) + eps)
        ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))
        return unspliced, spliced

    def sample(
        self,
    ) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        if self.decoder.linear_decoder is False:
            raise ValueError("Model not trained with linear decoder")
        w = self.decoder.rho_first_decoder.fc_layers[0][0].weight
        if self.use_batch_norm_decoder:
            bn = self.decoder.rho_first_decoder.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = w
        loadings = loadings.detach().cpu().numpy()

        return loadings
