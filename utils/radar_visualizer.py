import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import argparse


class RadarVisWindow:
    def __init__(self, data):
        self.window = gui.Application.instance.create_window(
            'Radar visualization', 3 * 128 + 50, 256)

        # Preprocessing -- record each slice
        self.im_views = []
        for axis in range(len(data.shape)):
            slices = []
            for i in range(data.shape[axis]):
                im = data.take(i, axis=axis).astype(np.float32)
                im = o3d.t.geometry.Image(o3c.Tensor(im)).colorize_depth(
                    10000, 0, 1).to_legacy()
                slices.append(im)
            self.im_views.append(slices)

        w = self.window
        em = w.theme.font_size
        spacing = int(np.round(0.25 * em))
        margins = gui.Margins(2 * spacing)

        self.panel = gui.Horiz(spacing, margins)

        axis_names = ['Elevation (top)', 'Azimuth (right)', 'Depth (front)']

        self.images = []

        def _on_slider(axis, i):
            self.images[axis].update_image(self.im_views[axis][int(i)])

        # Slider and image widget
        for axis in range(len(data.shape)):
            axis_panel = gui.Vert(spacing, margins)
            slider = gui.Slider(gui.Slider.INT)
            slider.set_limits(0, data.shape[axis] - 1)

            self.images.append(gui.ImageWidget())
            self.images[axis].update_image(self.im_views[axis][0])

            slider.set_on_value_changed(
                lambda i, axis=axis: _on_slider(axis, i))

            # Add child to the panel
            axis_panel.add_child(gui.Label(axis_names[axis]))
            axis_panel.add_child(slider)
            axis_panel.add_child(self.images[axis])

            # Add view panel to the main layout
            self.panel.add_child(axis_panel)

        w.add_child(self.panel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('radar_npz')

    args = parser.parse_args()
    radar_data = np.load(args.radar_npz)

    data = radar_data['intensity']

    app = gui.Application.instance
    app.initialize()
    w = RadarVisWindow(data)
    app.run()
