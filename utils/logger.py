import torch
from tensorboardX import SummaryWriter

class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            # assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)
    
    def update_group(self, group_dict, group_head='scalar', step=None, glob=False):
        """ Update a group of stats for specific head in TensorBoard """
        # head / name, respectively
        for k, v in group_dict.items():
            self.writer.add_scalar(group_head + "/" + k, v, self.step if step is None else step)
        
        # for better comparison within a run
        if glob:
            # head together
            self.writer.add_scalars('Global_'+group_head, group_dict)

    def flush(self):
        self.writer.flush()


# class WandbLogger(object):
#     def __init__(self, args):
#         self.args = args 
#         self._wandb = wandb

#         # Initialize a W&B run 
#         if self._wandb.run is None:
#             self._wandb.init(
#                 project=args.project,
#                 config=args)

#     def log_epoch_metrics(self, metrics, commit=True):
#         """
#         Log train/dev metrics onto W&B.
#         """
#         # Log number of model parameters as W&B summary
#         self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
#         metrics.pop('n_parameters', None)

#         # Log current epoch
#         self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
#         metrics.pop('epoch')

#         for k, v in metrics.items():
#             if 'train' in k:
#                 self._wandb.log({f'Global Train/{k}': v}, commit=False)
#             elif 'dev' in k:
#                 self._wandb.log({f'Global Dev/{k}': v}, commit=False)

#         self._wandb.log({})

#     def log_checkpoints(self):
#         output_dir = self.args.wandb_ckpt
#         model_artifact = self._wandb.Artifact(
#             self._wandb.run.id + "_model", type="model")

#         model_artifact.add_dir(output_dir)
#         self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

#     def set_steps(self):
#         # Set global training step
#         self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
#         # Set epoch-wise step
#         self._wandb.define_metric('Global Train/*', step_metric='epoch')
#         self._wandb.define_metric('Global Dev/*', step_metric='epoch')