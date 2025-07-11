from helpers.plot_helpers import Plot, Plot_Multiple

class PlotManager:
    def __init__(self, plots, n_random_starts=0):
        self.plots = {}
        self.plot_types = [plot["type"] for plot in plots.values()]
        for name, info in plots.items():
            if info["type"] == "single":
                self.plots[name] = Plot(
                    [], 
                    [], 
                    "Iteration", 
                    info["name"], 
                    log=(info["scale"] == "log-scale"),
                    shade_beginning=n_random_starts
                )
            elif info["type"] == "multiple":
                self.plots[name] = Plot_Multiple(
                    [[] for _ in range(len(info["legend"]))], 
                    [[] for _ in range(len(info["legend"]))], 
                    "Iteration", 
                    info["name"], 
                    shade_beginning=n_random_starts,
                    labels=info["legend"],
                    log=(info["scale"]=="log-scale"),
                    prioritize_first=info["prioritize_first"]
                )

    def add(self, iteration, data):
        for name, data in data.items():
            self.plots[name].add(iteration, data)

    def save(self, save_path):
        for i, (name, info) in enumerate(self.plots.items()):
            empty = False
            if self.plot_types[i] == "single" and self.plots[name].x == []:
                empty = True
            elif self.plot_types[i] == "multiple" and self.plots[name].x_lists == [[] for _ in range(len(self.plots[name].x_lists))]:
                empty = True

            if not empty:
                info.plot(f"{save_path}/{name}_iters.png")