# About
This repository contains an implementation of an interactive video display, that renders webcam input to the screen and applies image filters and geometric transformations to it.

# Build Instructions
1. Clone the repository
2. Set up OpenGL ([Linux](https://medium.com/geekculture/a-beginners-guide-to-setup-opengl-in-linux-debian-2bfe02ccd1e), [Windows](https://medium.com/swlh/setting-opengl-for-windows-d0b45062caf))
3. Install [OpenCV](https://opencv.org/get-started/)
4. Build using CMake

# Usage Instructions
The window displays the capture video with applied filter / transformations. Overlayed is the control panel, that allows for modification of all configuration values.

- **Backend Mode** (CPU, GPU): Should calculations be performed on CPU (using OpenCV) or on GPU (using OpenGL).
- **Image Filter** (None, Pencil, Retro): Which filter should be applied to the image.
- **Screen Width/Height** (int): To which resolution will the webcam input be up-/downscaled before processing.
- **Interactive Mode** (bool): Enable or disable interactive mode. In interactive mode, the image can be rotated (right click + drag), translated (left click + drag) or zoomed in/out (scroll mouse wheel).
- **FPS Display**: The current FPS is displayed here. Also, an average of the FPS can be tracked over a certain time window. The size of that window in seconds can be set in the number input. Tracking can be started using the button.
- **Save Image**: Save the currently rendered image to the "images" folder in the project root.
<img width="360" height="248" alt="config_overlay" src="https://github.com/user-attachments/assets/1550a85c-b417-4943-93da-3175302477f0" />
