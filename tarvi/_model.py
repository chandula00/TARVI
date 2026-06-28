"""TARVI model class."""
import logging
import warnings
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import devices_dsp, setup_anndata_dsp

from ._constants import REGISTRY_KEYS
from ._gnn import VelocityRefiner
from ._module import TARVIVAE
from ._utils import smooth_velocity

logger = logging.getLogger(__name__)


def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x)
    x_inv = torch.where(x > 20, x, x.expm1().log()).numpy()
    return x_inv


class TARVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """Transcription-Factor Aided RNA Velocity Inference.

    Extends VeloVI with TF-regulated transcription (ENCODE/ChEA databases),
    ODE residual supervision, and velocity confidence weighting.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~tarvi.TARVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    gamma_init_data
        Initialize gamma using the data-driven technique.
    linear_decoder
        Use a linear decoder from latent space to time.
    use_tf_regulation
        Whether to use TF-regulated transcription rates.
    tf_databases
        List of TF databases to use. Options: 'ENCODE', 'ChEA'.
    tf_data_dir
        Directory containing TF database files.
    tf_l1_weight
        L1 regularization weight for TF weights.
    **model_kwargs
        Keyword args for :class:`~tarvi.TARVIVAE`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        use_tf_regulation: bool = False,
        tf_databases: Optional[List[str]] = None,
        tf_data_dir: Optional[str] = None,
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
        gnn_k_neighbors: int = 15,
        # scVelo-inspired (best values from ablation study)
        ode_residual_weight: float = 1.0,
        velo_confidence_weight: float = 0.0,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.n_latent = n_latent

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        unspliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        alpha_unconstr = _softplus_inverse(us_upper)
        alpha_unconstr = np.asarray(alpha_unconstr).ravel()

        alpha_1_unconstr = np.zeros(us_upper.shape).ravel()
        lambda_alpha_unconstr = np.zeros(us_upper.shape).ravel()

        if gamma_init_data:
            gamma_unconstr = np.clip(_softplus_inverse(us_upper / ms_upper), None, 10)
        else:
            gamma_unconstr = None

        # === TF Regulation Setup (Phase 2) ===
        tf_target_mask = None
        n_tfs = 0
        tf_indices = None
        tf_init_weights = None
        self._tf_names = None

        if use_tf_regulation:
            from ._tf_module import build_tf_mask

            if tf_databases is None:
                tf_databases = ["ENCODE", "ChEA"]

            logger.info(f"Building TF mask from databases: {tf_databases}")
            tf_target_mask, tf_indices, tf_names, tf_init_weights = build_tf_mask(
                adata,
                databases=tf_databases,
                data_dir=tf_data_dir,
            )
            n_tfs = len(tf_names)
            self._tf_names = tf_names
            n_pairs = int(tf_target_mask.sum())
            logger.info(
                f"Found {n_tfs} TFs with {n_pairs} known regulatory pairs "
                f"across {adata.n_vars} genes"
            )

        self.module = TARVIVAE(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gamma_unconstr_init=gamma_unconstr,
            alpha_unconstr_init=alpha_unconstr,
            alpha_1_unconstr_init=alpha_1_unconstr,
            lambda_alpha_unconstr_init=lambda_alpha_unconstr,
            switch_spliced=ms_upper,
            switch_unspliced=us_upper,
            linear_decoder=linear_decoder,
            use_tf_regulation=use_tf_regulation,
            tf_target_mask=tf_target_mask,
            n_tfs=n_tfs,
            tf_indices=tf_indices,
            tf_init_weights=tf_init_weights,
            tf_l1_weight=tf_l1_weight,
            velocity_consistency_weight=velocity_consistency_weight,
            temporal_smoothness_weight=temporal_smoothness_weight,
            use_learned_time=use_learned_time,
            time_reg_weight=time_reg_weight,
            use_gnn_refinement=use_gnn_refinement,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_n_heads=gnn_n_heads,
            gnn_n_layers=gnn_n_layers,
            gnn_dropout=gnn_dropout,
            gnn_coherence_weight=gnn_coherence_weight,
            ode_residual_weight=ode_residual_weight,
            velo_confidence_weight=velo_confidence_weight,
            **model_kwargs,
        )

        # Build and register k-NN edge index for GNN refinement
        self._gnn_k_neighbors = gnn_k_neighbors
        if use_gnn_refinement and "connectivities" in adata.obsp:
            edge_index = self._build_edge_index_from_connectivities(
                adata.obsp["connectivities"], k=gnn_k_neighbors
            )
            self.module.set_edge_index(edge_index)
            logger.info(f"GNN: Built edge index with {edge_index.shape[1]} edges from connectivities")
        self._model_summary_string = (
            "TARVI Model with params: n_hidden={}, n_latent={}, n_layers={}, "
            "dropout_rate={}, use_tf={}, ode_res_w={}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            use_tf_regulation,
            ode_residual_weight,
        )
        self.init_params_ = self._get_init_params(locals())

    @staticmethod
    def _build_edge_index_from_connectivities(connectivities, k=15):
        """Build [2, E] edge index from sparse connectivity matrix.

        Takes the top-k neighbors per cell from the connectivities matrix.
        """
        import scipy.sparse as sp

        if sp.issparse(connectivities):
            conn = connectivities.tocsr()
        else:
            conn = sp.csr_matrix(connectivities)

        src_list, dst_list = [], []
        n_cells = conn.shape[0]
        for i in range(n_cells):
            row = conn[i]
            indices = row.indices
            data = row.data
            if len(indices) > k:
                # Take top-k by weight
                topk_idx = np.argsort(data)[-k:]
                indices = indices[topk_idx]
            for j in indices:
                src_list.append(j)
                dst_list.append(i)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        return edge_index

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: Optional[int] = 500,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        accelerator: str = "auto",
        devices: Union[int, list[int], str] = "auto",
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = True,
        gradient_clip_val: float = 10,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization
        weight_decay
            Weight decay for optimization
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping.
        gradient_clip_val
            Val for gradient clipping
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        user_plan_kwargs = plan_kwargs.copy() if isinstance(plan_kwargs, dict) else {}
        plan_kwargs = {"lr": lr, "weight_decay": weight_decay, "optimizer": "AdamW"}
        plan_kwargs.update(user_plan_kwargs)

        user_train_kwargs = trainer_kwargs.copy()
        trainer_kwargs = {"gradient_clip_val": gradient_clip_val}
        trainer_kwargs.update(user_train_kwargs)

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        # Add callback to update epoch fraction for curriculum NB
        from lightning.pytorch.callbacks import Callback

        class EpochFracCallback(Callback):
            def __init__(self, module, max_epochs):
                self._module = module
                self._max_epochs = max_epochs

            def on_train_epoch_start(self, trainer, pl_module):
                frac = trainer.current_epoch / max(self._max_epochs, 1)
                self._module._current_epoch_frac = frac

        callbacks = trainer_kwargs.get("callbacks", []) or []
        callbacks.append(EpochFracCallback(self.module, max_epochs))
        trainer_kwargs["callbacks"] = callbacks

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_state_assignment(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        hard_assignment: bool = False,
        n_samples: int = 20,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
        """Returns cells by genes by states probabilities."""
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        states = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats

    @torch.inference_mode()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        time_statistic: Literal["mean", "max"] = "mean",
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns the cells by genes latent time."""
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.module.t_max - switch_time)
                )

                if time_statistic == "mean":
                    output = (
                        ind_prob * ind_time
                        + rep_prob * rep_time
                        + steady_prob * switch_time
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return times

    @torch.inference_mode()
    def get_learned_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: int = 25,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Returns per-cell learned latent time from the time head.

        Only available when use_learned_time=True.
        Returns array of shape [n_cells] with values in [0, 1].
        """
        if not self.module.use_learned_time:
            raise ValueError("Model was not trained with use_learned_time=True")

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        all_times = []
        for tensors in scdl:
            sample_times = []
            for _ in range(n_samples):
                inference_outputs, _ = self.module.forward(
                    tensors=tensors, compute_loss=False,
                )
                t = inference_outputs["learned_time"]  # [batch]
                sample_times.append(t.cpu().numpy())
            # Mean over samples
            all_times.append(np.mean(sample_times, axis=0))

        return np.concatenate(all_times, axis=0)

    @torch.inference_mode()
    def get_velocity(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal["spliced", "unspliced"] = "spliced",
        clip: bool = True,
        smooth: bool = False,
        smooth_alpha: float = 0.5,
        refine: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use.
        gene_list
            Return velocities for a subset of genes.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return.
        batch_size
            Minibatch size for data loading into model.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a numpy array instead of a DataFrame.
        velo_statistic
            Whether to compute expected velocity over states, or MAP velocity.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value.
        smooth
            Whether to apply kNN velocity smoothing (Phase 4).
        smooth_alpha
            Mixing weight for smoothing. 0 = no smoothing, 1 = full neighbor average.

        Returns
        -------
        Velocity estimates.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
            n_samples = 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        velos = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                beta = inference_outputs["beta"]
                gamma = inference_outputs["gamma"]
                tau = generative_outputs["px_tau"]
                rho = generative_outputs["px_rho"]

                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = switch_time * rho
                u_0, s_0 = self.module._get_induction_unspliced_spliced(
                    alpha, alpha_1, lambda_alpha, beta, gamma, switch_time
                )
                rep_time = (self.module.t_max - switch_time) * tau
                mean_u_rep, mean_s_rep = self.module._get_repression_unspliced_spliced(
                    u_0,
                    s_0,
                    beta,
                    gamma,
                    rep_time,
                )
                if velo_mode == "spliced":
                    velo_rep = beta * mean_u_rep - gamma * mean_s_rep
                else:
                    velo_rep = -beta * mean_u_rep
                mean_u_ind, mean_s_ind = self.module._get_induction_unspliced_spliced(
                    alpha, alpha_1, lambda_alpha, beta, gamma, ind_time
                )
                if velo_mode == "spliced":
                    velo_ind = beta * mean_u_ind - gamma * mean_s_ind
                else:
                    transcription_rate = alpha_1 - (alpha_1 - alpha) * torch.exp(
                        -lambda_alpha * ind_time
                    )
                    velo_ind = transcription_rate - beta * mean_u_ind

                if velo_mode == "spliced":
                    velo_steady = torch.zeros_like(velo_ind)
                else:
                    velo_steady = torch.zeros_like(velo_ind)

                # expectation
                if velo_statistic == "mean":
                    output = (
                        ind_prob * velo_ind
                        + rep_prob * velo_rep
                        + steady_prob * velo_steady
                    )
                # maximum
                else:
                    v = torch.stack(
                        [
                            velo_ind,
                            velo_steady.expand(velo_ind.shape),
                            velo_rep,
                            torch.zeros_like(velo_rep),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (v * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)

        if clip:
            velos = np.clip(velos, -spliced[indices], None)

        # GNN velocity refinement (applied on full dataset)
        if refine and self.module.use_gnn_refinement and self.module.velocity_refiner is not None:
            velos = self._apply_gnn_refinement(adata, velos, indices)

        # Phase 4: Optional kNN velocity smoothing
        if smooth and "connectivities" in adata.obsp:
            conn = adata.obsp["connectivities"]
            if indices is not None:
                # Subset connectivity matrix for the selected cells
                conn = conn[indices][:, indices]
            velos = smooth_velocity(velos, conn, alpha=smooth_alpha)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return velos

    def get_velocity_pseudotime(
        self,
        adata=None,
        vkey: str = "velocity",
        blend: float = 0.5,
        n_samples: int = 25,
    ) -> np.ndarray:
        """Compute velocity-graph pseudotime (scVelo-inspired).

        Builds a directed velocity transition graph from TARVI velocities,
        then runs diffusion pseudotime (VPT) on it — same as scVelo's
        latent_time computation. Blends with the learned time head output.

        Parameters
        ----------
        adata
            AnnData with velocity graph already computed (scv.tl.velocity_graph called).
        vkey
            Velocity key in adata.layers.
        blend
            Weight for velocity-graph pseudotime vs learned time head.
            blend=1.0 → pure velocity-graph PT (like scVelo)
            blend=0.0 → pure learned time head
        n_samples
            Samples for learned time estimation.

        Returns
        -------
        latent_time : np.ndarray [n_cells]  in [0, 1]
        """
        import scvelo as scv
        from scipy.sparse import issparse

        adata = self._validate_anndata(adata)

        # Step 1: Get velocity-graph pseudotime via scVelo
        if f"{vkey}_graph" not in adata.uns:
            logger.info("Computing velocity graph for pseudotime...")
            scv.tl.velocity_graph(adata, vkey=vkey)

        logger.info("Computing velocity pseudotime (scVelo-style diffusion)...")
        scv.tl.velocity_pseudotime(adata, vkey=vkey)
        vpt = adata.obs["velocity_pseudotime"].values.copy()

        # Normalize to [0,1]
        vpt_min, vpt_max = np.nanmin(vpt), np.nanmax(vpt)
        vpt_norm = (vpt - vpt_min) / (vpt_max - vpt_min + 1e-8)
        vpt_norm = np.nan_to_num(vpt_norm, nan=0.5)

        if blend < 1.0 and self.module.use_learned_time:
            # Step 2: Learned time from time head
            learned_t = self.get_learned_time(adata=adata, n_samples=n_samples)

            # Align direction to velocity pseudotime
            from scipy.stats import spearmanr
            corr, _ = spearmanr(learned_t, vpt_norm)
            if corr < 0:
                learned_t = 1.0 - learned_t

            # Blend: velocity-graph PT (biologically directed) + learned time (smooth)
            latent_time = blend * vpt_norm + (1.0 - blend) * learned_t
        else:
            latent_time = vpt_norm

        # Normalize final
        t_min, t_max = latent_time.min(), latent_time.max()
        latent_time = (latent_time - t_min) / (t_max - t_min + 1e-8)
        return latent_time

    @torch.inference_mode()
    def _apply_gnn_refinement(self, adata, velos, indices):
        """Apply trained GNN velocity refiner on the full velocity matrix.

        Builds a k-NN edge index for the selected cells, gets z embeddings,
        and runs the VelocityRefiner.
        """
        # Get latent z for the selected cells
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=256)
        z_list = []
        for tensors in scdl:
            inference_outputs, _ = self.module.forward(tensors=tensors, compute_loss=False)
            z_list.append(inference_outputs["z"].cpu())
        z_all = torch.cat(z_list, dim=0)  # [N, z_dim]

        # Build edge index for these cells
        n_cells = z_all.shape[0]
        k = min(self._gnn_k_neighbors, n_cells - 1)
        dists = torch.cdist(z_all, z_all)
        dists.fill_diagonal_(float('inf'))
        _, nn_idx = dists.topk(k, largest=False)
        src = nn_idx.reshape(-1)
        dst = torch.arange(n_cells).unsqueeze(1).expand(-1, k).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)

        # Move to model device
        device = next(self.module.parameters()).device
        velocity_tensor = torch.tensor(velos, dtype=torch.float32, device=device)
        z_all = z_all.to(device)
        edge_index = edge_index.to(device)

        # Apply refiner
        refined, gate = self.module.velocity_refiner(velocity_tensor, z_all, edge_index)
        logger.info(f"GNN refinement: mean gate={gate.mean().item():.3f}")

        return refined.cpu().numpy()

    @torch.inference_mode()
    def get_expression_fit(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        restrict_to_latent_dim: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the fitted spliced and unspliced abundance (s(t) and u(t))."""
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        fits_s = []
        fits_u = []
        for tensors in scdl:
            minibatch_samples_s = []
            minibatch_samples_u = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                    generative_kwargs={"latent_dim": restrict_to_latent_dim},
                )

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (
                    mixture_dist_s,
                    mixture_dist_u,
                    _,
                ) = self.module.get_px(
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
                fit_s = mixture_dist_s.mean
                fit_u = mixture_dist_u.mean

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u

    @torch.inference_mode()
    def get_gene_likelihood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the likelihood per gene. Higher is better."""
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        rls = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                spliced = tensors[REGISTRY_KEYS.X_KEY]
                unspliced = tensors[REGISTRY_KEYS.U_KEY]

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (
                    mixture_dist_s,
                    mixture_dist_u,
                    _,
                ) = self.module.get_px(
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
                device = gamma.device
                reconst_loss_s = -mixture_dist_s.log_prob(spliced.to(device))
                reconst_loss_u = -mixture_dist_u.log_prob(unspliced.to(device))
                output = -(reconst_loss_s + reconst_loss_u)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            rls.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rls[-1] = np.mean(rls[-1], axis=0)

        rls = np.concatenate(rls, axis=0)
        return rls

    @torch.inference_mode()
    def get_rates(self):
        """Get kinetic rates.

        For TF-regulated model, returns the basal alpha (not cell-specific).
        Use get_tf_weights() for TF-specific contributions.
        """
        if self.module.use_tf_regulation and self.module.tf_module is not None:
            gamma = torch.clamp(F.softplus(self.module.gamma_mean_unconstr), 0, 50)
            beta = torch.clamp(F.softplus(self.module.beta_mean_unconstr), 0, 50)
            alpha = F.softplus(self.module.tf_module.b_unconstr)
            alpha_1 = self.module.alpha_1_unconstr
            lambda_alpha = self.module.lambda_alpha_unconstr
        else:
            gamma, beta, alpha, alpha_1, lambda_alpha = self.module._get_rates()

        return {
            "beta": beta.cpu().numpy(),
            "gamma": gamma.cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
            "alpha_1": alpha_1.cpu().numpy(),
            "lambda_alpha": lambda_alpha.cpu().numpy(),
        }

    @torch.inference_mode()
    def get_tf_weights(self) -> Optional[pd.DataFrame]:
        """Get learned TF-target regulatory weights.

        Returns
        -------
        DataFrame with genes as rows and TFs as columns, containing learned weights.
        Only includes non-zero (masked) entries. Returns None if TF regulation is not used.
        """
        if not self.module.use_tf_regulation or self.module.tf_module is None:
            return None

        weights = self.module.tf_module.get_weights().numpy()
        adata = self.adata_manager.adata

        df = pd.DataFrame(
            weights,
            index=adata.var_names,
            columns=self._tf_names,
        )
        return df

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer: str,
        unspliced_layer: str,
        **kwargs,
    ) -> Optional[AnnData]:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        spliced_layer
            Layer in adata with spliced normalized expression
        unspliced_layer
            Layer in adata with unspliced normalized expression.

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, spliced_layer, is_count_data=False),
            LayerField(REGISTRY_KEYS.U_KEY, unspliced_layer, is_count_data=False),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def get_directional_uncertainty(
        self,
        adata: Optional[AnnData] = None,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
    ):
        adata = self._validate_anndata(adata)

        logger.info("Sampling from model...")
        velocities_all = self.get_velocity(
            n_samples=n_samples, return_mean=False, gene_list=gene_list
        )

        df, cosine_sims = _compute_directional_statistics_tensor(
            tensor=velocities_all, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        return df, cosine_sims

    def get_permutation_scores(
        self, labels_key: str, adata: Optional[AnnData] = None
    ) -> Tuple[pd.DataFrame, AnnData]:
        """Compute permutation scores."""
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if labels_key not in adata.obs:
            raise ValueError(f"{labels_key} not found in adata.obs")

        bdata = self._shuffle_layer_celltype(
            adata_manager, labels_key, REGISTRY_KEYS.X_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)
        bdata = self._shuffle_layer_celltype(
            bdata_manager, labels_key, REGISTRY_KEYS.U_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)

        ms_ = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_ = adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        ms_p = bdata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_p = bdata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        spliced_, unspliced_ = self.get_expression_fit(adata, n_samples=10)
        root_squared_error = np.abs(spliced_ - ms_)
        root_squared_error += np.abs(unspliced_ - mu_)

        spliced_p, unspliced_p = self.get_expression_fit(bdata, n_samples=10)
        root_squared_error_p = np.abs(spliced_p - ms_p)
        root_squared_error_p += np.abs(unspliced_p - mu_p)

        celltypes = np.unique(adata.obs[labels_key])

        dynamical_df = pd.DataFrame(
            index=adata.var_names,
            columns=celltypes,
            data=np.zeros((adata.shape[1], len(celltypes))),
        )
        N = 200
        for ct in celltypes:
            for g in adata.var_names.tolist():
                x = root_squared_error_p[g][adata.obs[labels_key] == ct]
                y = root_squared_error[g][adata.obs[labels_key] == ct]
                ratio = ttest_ind(x[:N], y[:N])[0]
                dynamical_df.loc[g, ct] = ratio

        return dynamical_df, bdata

    def _shuffle_layer_celltype(
        self, adata_manager: AnnDataManager, labels_key: str, registry_key: str
    ) -> AnnData:
        """Shuffle cells within cell types for each gene."""
        from scvi.data._constants import _SCVI_UUID_KEY

        bdata = adata_manager.adata.copy()
        labels = bdata.obs[labels_key]
        del bdata.uns[_SCVI_UUID_KEY]
        self._validate_anndata(bdata)
        bdata_manager = self.get_anndata_manager(bdata)

        unspliced = bdata_manager.get_from_registry(registry_key)
        u_registry = bdata_manager.data_registry[registry_key]
        attr_name = u_registry.attr_name
        attr_key = u_registry.attr_key

        for lab in np.unique(labels):
            mask = np.asarray(labels == lab)
            unspliced_ct = unspliced[mask].copy()
            unspliced_ct = np.apply_along_axis(
                np.random.permutation, axis=0, arr=unspliced_ct
            )
            unspliced[mask] = unspliced_ct
        if attr_key is None:
            setattr(bdata, attr_name, unspliced)
        elif attr_key is not None:
            attribute = getattr(bdata, attr_name)
            attribute[attr_key] = unspliced
            setattr(bdata, attr_name, attribute)

        return bdata


def _compute_directional_statistics_tensor(
    tensor: np.ndarray, n_jobs: int, n_cells: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n_cells))
    df["directional_variance"] = np.nan
    df["directional_difference"] = np.nan
    df["directional_cosine_sim_variance"] = np.nan
    df["directional_cosine_sim_difference"] = np.nan
    df["directional_cosine_sim_mean"] = np.nan
    logger.info("Computing the uncertainties...")
    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_directional_statistics_per_cell)(tensor[:, cell_index, :])
        for cell_index in range(n_cells)
    )
    cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
    df.loc[:, "directional_cosine_sim_variance"] = [
        results[i][1] for i in range(n_cells)
    ]
    df.loc[:, "directional_cosine_sim_difference"] = [
        results[i][2] for i in range(n_cells)
    ]
    df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
    df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
    df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

    return df, cosine_sims


def _directional_statistics_per_cell(
    tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal function for parallelization."""
    n_samples = tensor.shape[0]
    mean_velocity_of_cell = tensor.mean(0)
    cosine_sims = [
        _cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples)
    ]
    angle_samples = [np.arccos(el) for el in cosine_sims]
    return (
        cosine_sims,
        np.var(cosine_sims),
        np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
        np.var(angle_samples),
        np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
        np.mean(cosine_sims),
    )


def _centered_unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the centered unit vector of the vector."""
    vector = vector - np.mean(vector)
    return vector / np.linalg.norm(vector)


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns cosine similarity of the vectors."""
    v1_u = _centered_unit_vector(v1)
    v2_u = _centered_unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
