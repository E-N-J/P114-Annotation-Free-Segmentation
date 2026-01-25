import matplotlib.pyplot as plt
from utils import get_environment
from matplotlib.widgets import Button as mplButton

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except ImportError:
    widgets = None
    
def visualise_results(X_input, L_rpca, S_rpca, model_results):
    """
    Visualises the original, RPCA decomposed, and deep model decomposed images.
    Auto-detects environment to serve either IPyWidgets (Notebook) or Matplotlib Widgets (Script).
    """
    batch_size = X_input.shape[0]
    env = get_environment()
    is_notebook = (env == 'notebook' or env == 'colab')

    state = {'idx': 0}

    def generate_plot(idx):
        fig, ax = plt.subplots(
            2 + len(model_results), 3, 
            figsize=(10, 2 * (2 + len(model_results))), 
            constrained_layout=True
        )
        
        def show_img(ax_idx, img, title):
            ax_idx.imshow(img.squeeze(), cmap='gray')
            ax_idx.set_title(title)
            ax_idx.axis('off')

        # Original
        show_img(ax[0,0], X_input[idx], f"Input [{idx}]")
        ax[0,1].axis('off'); ax[0,2].axis('off')

        # RPCA
        show_img(ax[1,0], L_rpca[idx], "RPCA L (Background)")
        show_img(ax[1,1], abs(S_rpca[idx]), "RPCA S (Anomalies)")
        ax[1,2].axis('off')

        # Deep Models
        for i, (name, (L, S)) in enumerate(model_results.items()):
            row = 2 + i
            show_img(ax[row,0], L[idx], f"{name} L")
            show_img(ax[row,1], abs(S[idx]), f"{name} S")
            ax[row,2].axis('off')
            
        return fig

    if is_notebook:
        if widgets is None:
            print("Warning: ipywidgets not installed. Visualising first batch only.")
            fig = generate_plot(0)
            plt.show()
            return

        out = widgets.Output()

        btn_prev = widgets.Button(description="Previous", icon="arrow-left")
        btn_next = widgets.Button(description="Next", icon="arrow-right")

        def on_click_next(b):
            state['idx'] = (state['idx'] + 1) % batch_size
            render()
            
        def on_click_prev(b):
            state['idx'] = (state['idx'] - 1) % batch_size
            render()

        def render():
            plt.ioff() # Prevent auto-display
            fig = generate_plot(state['idx'])
            plt.ion()  # Restore auto-display

            with out:
                clear_output(wait=True)
                display(fig)

            plt.close(fig)

        btn_prev.on_click(on_click_prev)
        btn_next.on_click(on_click_next)
        
        render()
        
        ui = widgets.HBox([btn_prev, btn_next])
        display(widgets.VBox([ui, out]))
        
    else:
        fig = generate_plot(0)
        
        def update_script_view(idx):
            ax_list = fig.axes
            
            for a in ax_list:
                if a not in [ax_prev, ax_next]:
                    a.clear()
            
            axs = fig.subplots(2 + len(model_results), 3) if not fig.axes else fig.axes

            fig.clf() 
            
            gs = fig.add_gridspec(2 + len(model_results), 3)
            axs = gs.subplots()
            
            def show_img_s(ax_idx, img, title):
                ax_idx.imshow(img.squeeze(), cmap='gray')
                ax_idx.set_title(title)
                ax_idx.axis('off')

            show_img_s(axs[0,0], X_input[idx], f"Input [{idx}]")
            axs[0,1].axis('off'); axs[0,2].axis('off')

            show_img_s(axs[1,0], L_rpca[idx], "RPCA L")
            show_img_s(axs[1,1], abs(S_rpca[idx]), "RPCA S")
            axs[1,2].axis('off')

            for i, (name, (L, S)) in enumerate(model_results.items()):
                r = 2 + i
                show_img_s(axs[r,0], L[idx], f"{name} L")
                show_img_s(axs[r,1], abs(S[idx]), f"{name} S")
                axs[r,2].axis('off')
            
            add_buttons()
            fig.canvas.draw_idle()

        ax_prev = None
        ax_next = None
        b_prev = None
        b_next = None

        def next_click(event):
            state['idx'] = (state['idx'] + 1) % batch_size
            update_script_view(state['idx'])

        def prev_click(event):
            state['idx'] = (state['idx'] - 1) % batch_size
            update_script_view(state['idx'])

        def add_buttons():
            nonlocal ax_prev, ax_next, b_prev, b_next
            ax_prev = fig.add_axes([0.3, 0.02, 0.1, 0.05])
            ax_next = fig.add_axes([0.6, 0.02, 0.1, 0.05])
            b_prev = mplButton(ax_prev, 'Previous')
            b_next = mplButton(ax_next, 'Next')
            b_prev.on_clicked(prev_click)
            b_next.on_clicked(next_click)

        add_buttons()
        plt.show()
