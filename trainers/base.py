import matplotlib.pyplot as plt
from IPython.display import display
from utils import get_environment

class BaseTrainer:
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            # Fallback for models with no parameters (e.g., RPCA)
            self.device = next(model.buffers()).device
            
        print(f"Training with {self.device}")
        self.global_step = 0
        self.histories = {}
        
        env = get_environment()
        self.is_notebook = (env == 'notebook' or env == 'colab')
        
        if self.is_notebook:
            from tqdm.notebook import tqdm
            self.tqdm = tqdm
        else:
            from tqdm import tqdm
            self.tqdm = tqdm
        
        self.live_fig = None
        self.live_axes = None
        self.display_handle = None

    def _create_figure(self, num_plots):
        """Helper to centralise figure creation."""
        fig, axes = plt.subplots(
            1, num_plots, 
            figsize=(8 * num_plots, 5), 
            constrained_layout=True
        )
        if num_plots == 1: axes = [axes]
        return fig, axes

    def plot_metrics(self, show_figure=True, log_scale=False, save_path='training_curve.png'):
        if not self.histories:
            print("No training history found.")
            return

        num_plots = 2 if log_scale else 1

        if self.is_notebook:
            if self.live_fig is None:
                self.live_fig, self.live_axes = self._create_figure(num_plots)
            
            fig, axes = self.live_fig, self.live_axes
            for ax in axes: ax.clear()
        else:
            fig, axes = self._create_figure(num_plots)

        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        
        for plot_idx, ax in enumerate(axes):
            for idx, (metric_name, metric_values) in enumerate(self.histories.items()):
                if metric_values:
                    color = colors[idx % len(colors)]
                    label = metric_name.replace('_', ' ').title()
                    linestyle = '-' if metric_name == 'objective' else '--'
                    ax.plot(metric_values, linestyle=linestyle, color=color, label=label)
            
            ax.set_title('Training Metrics')
            ax.set_xlabel('Step/Epoch')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
            
            if log_scale and plot_idx == 1:
                ax.set_yscale('log')
                ax.set_title('Training Metrics (Log Scale)')

        if self.is_notebook:
            if self.display_handle is None:
                self.display_handle = display(self.live_fig, display_id=True)
            else:
                self.display_handle.update(self.live_fig)
        else:
            plt.savefig(save_path, dpi=300)
            if show_figure:
                plt.show()
            else:
                plt.close(fig)
                
