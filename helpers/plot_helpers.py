import matplotlib.pyplot as plt

class Plot:
    def __init__(self, x, y, x_label, y_label, log=False, shade_beginning=None, alpha=1.0):
        self.x = x
        self.y = y
        self.x_label = x_label
        self.y_label = y_label
        self.alpha = alpha
        self.log = log
        self.shade_beginning = shade_beginning

    def add(self, x_val, y_val):
        self.x.append(x_val)
        self.y.append(y_val)

    def plot(self, save_path):
        plt.close("all")
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, alpha=self.alpha)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if self.log:
            ax.set_yscale("log")
        if self.shade_beginning is not None:
            ax.axvspan(0, self.shade_beginning, color='gray', alpha=0.2)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(save_path)
        plt.close(fig)


class Plot_Multiple:
    def __init__(self, y_lists, x_lists, x_label, y_label, log=False, prioritize_first=False, labels=None, shade_beginning=None):
        self.y_lists = y_lists
        self.x_lists = x_lists
        self.x_label = x_label
        self.y_label = y_label
        self.labels = labels if labels is not None else [f"Line {i+1}" for i in range(len(y_lists))]
        self.shade_beginning = shade_beginning
        self.log = log
        self.prioritize_first = prioritize_first

    def add(self, iteration, data):
        for i, label in enumerate(self.labels):
            if label in data.keys():
                self.x_lists[i].append(iteration)
                self.y_lists[i].append(data[label])


    def plot(self, save_path):
        plt.close("all")
        fig, ax = plt.subplots()
        
        for i, (x_list, y_list, label) in enumerate(zip(self.x_lists, self.y_lists, self.labels)):
            alpha = 0.6 if not self.prioritize_first else (1 if i == 0 else 0.4)
            ax.plot(x_list, y_list, label=label, alpha=alpha)
        if self.log:
            ax.set_yscale("log")
        if self.shade_beginning is not None:
            ax.axvspan(0, self.shade_beginning, color='gray', alpha=0.2)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.legend(loc="upper left", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(save_path)
        plt.close(fig)