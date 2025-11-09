import matplotlib.pyplot as plt

# Use IPython display utilities when available (notebook); fall back to plt.pause for scripts
try:
    from IPython.display import clear_output, display
    _use_ipython = True
except Exception:
    _use_ipython = False

plt.ion()

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='mean score')
    plt.legend()

    if _use_ipython:
        clear_output(wait=True)
        display(plt.gcf())
    else:
        # for normal terminals / scripts
        plt.draw()
        plt.pause(0.001)