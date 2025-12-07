import math
import sys

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QOpenGLWidget,
                             QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout, QMessageBox,
                             QGroupBox, QGridLayout, QDoubleSpinBox)


class GeodesicSphere(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initial_rotation_y = 0
        self.initial_rotation_x = 0
        self.initial_rotation_z = 0
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0  # Added Z-axis rotation
        self.last_pos = None
        self.points = []
        self.geodesics = []
        self.sphere_radius = 1.0

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h != 0 else 1, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Position camera
        glTranslatef(0.0, 0.0, -4.0)

        # Apply rotations in ZYX order (or YXZ depending on what feels natural)
        # For intuitive control: Z first, then Y, then X
        glRotatef(self.rotation_z, 0, 0, 1)  # Z-axis rotation
        glRotatef(self.rotation_y, 0, 1, 0)  # Y-axis rotation
        glRotatef(self.rotation_x, 1, 0, 0)  # X-axis rotation

        # Draw sphere with wireframe
        glColor3f(0.3, 0.5, 0.8)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        quadric = gluNewQuadric()
        gluSphere(quadric, self.sphere_radius, 50, 50)
        gluDeleteQuadric(quadric)

        # Draw wireframe overlay
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.3, 0.4)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.0)
        quadric = gluNewQuadric()
        gluSphere(quadric, self.sphere_radius + 0.001, 20, 20)
        gluDeleteQuadric(quadric)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Draw geodesics
        glLineWidth(3.0)
        glColor3f(1.0, 0.8, 0.0)
        for geodesic in self.geodesics:
            glBegin(GL_LINE_STRIP)
            for point in geodesic:
                glVertex3fv(point)
            glEnd()

        # Draw selected points
        glPointSize(10.0)
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_POINTS)
        for point in self.points:
            glVertex3fv(point)
        glEnd()

        glEnable(GL_LIGHTING)
        glLineWidth(1.0)
        glPointSize(1.0)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.last_pos = event.pos()
            self.initial_rotation_z = self.rotation_z

        elif event.button() == Qt.LeftButton:
            self.last_pos = event.pos()

            self.initial_rotation_x = self.rotation_x
            self.initial_rotation_y = self.rotation_y

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return

        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if event.buttons() & Qt.RightButton:
            self.rotation_z = self.initial_rotation_z + dx * 0.5
            self.update()

        elif event.buttons() & Qt.LeftButton:
            self.rotation_x = self.initial_rotation_x + dy * 0.5
            self.rotation_y = self.initial_rotation_y + dx * 0.5
            self.update()

    def mouseReleaseEvent(self, event):
        self.last_pos = None

    def add_point_from_coordinates(self, lon, lat):
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)

        x = math.cos(lat_rad) * math.cos(lon_rad) * self.sphere_radius
        y = math.cos(lat_rad) * math.sin(lon_rad) * self.sphere_radius
        z = math.sin(lat_rad) * self.sphere_radius

        point = (x, y, z)
        self.points.append(point)

        if len(self.points) >= 2:
            p1 = self.points[-2]
            p2 = self.points[-1]
            geodesic = self.compute_geodesic(p1, p2)
            self.geodesics.append(geodesic)

        self.update()
        return True, f"Point added at Longitude: {lon}°, Latitude: {lat}°\nCartesian: ({x:.3f}, {y:.3f}, {z:.3f})"

    def compute_geodesic(self, p1, p2, num_points=100):
        p1 = np.array(p1) / np.linalg.norm(p1)
        p2 = np.array(p2) / np.linalg.norm(p2)

        dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
        angle = np.arccos(dot)

        if abs(dot + 1.0) < 1e-10:  # Antipodal points (dot ≈ -1)
            if abs(p1[0]) > 1e-10 or abs(p1[1]) > 1e-10:
                axis = np.cross(p1, [0, 0, 1])
            else:
                axis = np.cross(p1, [1, 0, 0])
            axis = axis / np.linalg.norm(axis)

            geodesic = []
            for i in range(num_points):
                t = i / (num_points - 1)
                theta = t * np.pi

                # Rodrigues' rotation formula
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                point = (p1 * cos_theta +
                         np.cross(axis, p1) * sin_theta +
                         axis * np.dot(axis, p1) * (1 - cos_theta))

                point = point * self.sphere_radius
                geodesic.append(point)
            return geodesic

        geodesic = []
        for i in range(num_points):
            t = i / (num_points - 1)

            # Spherical linear interpolation (slerp)
            if angle > 0.001:
                s1 = np.sin((1 - t) * angle) / np.sin(angle)
                s2 = np.sin(t * angle) / np.sin(angle)
                point = s1 * p1 + s2 * p2
            else:
                # Points are very close, use linear interpolation
                point = (1 - t) * p1 + t * p2

            norm = np.linalg.norm(point)
            if norm > 1e-10:
                point = point / norm * self.sphere_radius
            else:
                point = np.array([1e-10, 0, 0]) * self.sphere_radius

            geodesic.append(point)

        return geodesic

    def clear_all_points(self):
        self.points.clear()
        self.geodesics.clear()
        self.update()


class ControlPanel(QWidget):
    def __init__(self, gl_widget, parent=None):
        super().__init__(parent)
        self.clear_button = QPushButton("Clear All Points")
        self.rotation_label = QLabel("Rotation: X=0°, Y=0°, Z=0°")
        self.reset_rotation_button = QPushButton("Reset Rotation")
        self.status_label = QLabel("Ready")
        self.add_button = QPushButton("Add Point")
        self.lat_input = QDoubleSpinBox()
        self.lon_input = QDoubleSpinBox()
        self.gl_widget = gl_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        instructions_group = QGroupBox("Controls & Instructions")
        instructions_layout = QVBoxLayout()

        instructions = QLabel(
            "Mouse Controls:\n"
            "• Right drag: Rotate around Z-axis (roll)\n"
            "• Left drag: Rotate around X/Y axes\n\n"
            "Coordinates:\n"
            "Longitude: -180° to 180° (0° = Prime Meridian)\n"
            "Latitude: -90° to 90° (0° = Equator)"
        )
        instructions.setWordWrap(True)
        instructions.setMaximumWidth(250)
        instructions_layout.addWidget(instructions)
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)

        input_group = QGroupBox("Add Point (Geographic Coordinates)")
        input_layout = QVBoxLayout()

        lon_layout = QHBoxLayout()
        lon_layout.addWidget(QLabel("Longitude:"))
        self.lon_input.setRange(-180.0, 180.0)
        self.lon_input.setDecimals(2)
        self.lon_input.setSingleStep(1.0)
        self.lon_input.setValue(0.0)
        self.lon_input.setSuffix("°")
        self.lon_input.setMaximumWidth(120)
        lon_layout.addWidget(self.lon_input)
        lon_layout.addStretch()
        input_layout.addLayout(lon_layout)

        lat_layout = QHBoxLayout()
        lat_layout.addWidget(QLabel("Latitude:"))
        self.lat_input.setRange(-90.0, 90.0)
        self.lat_input.setDecimals(2)
        self.lat_input.setSingleStep(1.0)
        self.lat_input.setValue(0.0)
        self.lat_input.setSuffix("°")
        self.lat_input.setMaximumWidth(120)
        lat_layout.addWidget(self.lat_input)
        lat_layout.addStretch()
        input_layout.addLayout(lat_layout)

        self.add_button.clicked.connect(self.add_point)
        input_layout.addWidget(self.add_button)

        self.status_label.setMaximumHeight(60)
        self.status_label.setWordWrap(True)
        input_layout.addWidget(self.status_label)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        preset_group = QGroupBox("Preset Locations")
        preset_layout = QGridLayout()

        # lon, lat
        presets = [
            ("North Pole", (0.0, 90.0)),
            ("South Pole", (0.0, -90.0)),
            ("(0,0)", (0.0, 0.0)),
            ("Boston", (-71.05, 42.4)),
            ("Tokyo", (139.7, 35.7)),
            ("Astana", (71.4272, 51.1655)),
            ("Bucharest", (26.1025, 44.4268)),
            ("Washington DC", (-77.0369, 38.9072))
        ]

        for i, (location_name, coords) in enumerate(presets):
            btn = QPushButton(location_name)
            btn.setMaximumWidth(110)
            btn.clicked.connect(lambda checked, c=coords: self.set_preset_coords(c))
            row = i // 2
            col = i % 2
            preset_layout.addWidget(btn, row, col)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        rotation_group = QGroupBox("Rotation Controls")
        rotation_layout = QVBoxLayout()

        self.reset_rotation_button.clicked.connect(self.reset_rotation)
        rotation_layout.addWidget(self.reset_rotation_button)

        rotation_layout.addWidget(self.rotation_label)

        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)

        self.clear_button.clicked.connect(self.clear_points)
        layout.addWidget(self.clear_button)

        layout.addStretch(1)

    def set_preset_coords(self, coords):
        lon, lat = coords
        self.lon_input.setValue(lon)
        self.lat_input.setValue(lat)

    def add_point(self):
        lon = self.lon_input.value()
        lat = self.lat_input.value()

        success, message = self.gl_widget.add_point_from_coordinates(lon, lat)

        if success:
            self.status_label.setText(message)
            self.update_rotation_display()
        else:
            QMessageBox.warning(self, "Error", message)

    def clear_points(self):
        self.gl_widget.clear_all_points()
        self.status_label.setText("All points cleared")

    def reset_rotation(self):
        self.gl_widget.rotation_x = 0
        self.gl_widget.rotation_y = 0
        self.gl_widget.rotation_z = 0
        self.gl_widget.update()
        self.update_rotation_display()
        self.status_label.setText("Rotation reset to initial position")

    def update_rotation_display(self):
        rotation_text = f"Rotation: X={self.gl_widget.rotation_x:.1f}°, Y={self.gl_widget.rotation_y:.1f}°, Z={self.gl_widget.rotation_z:.1f}°"
        self.rotation_label.setText(rotation_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geodesic Sphere Visualizer - Full 3D Rotation")
        self.setGeometry(100, 100, 950, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.gl_widget = GeodesicSphere()

        self.control_panel = ControlPanel(self.gl_widget)
        self.control_panel.setMaximumWidth(320)

        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.gl_widget, 1)  # Give sphere most of the space

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.gl_widget.clear_all_points()
            self.control_panel.status_label.setText("All points cleared")
        elif event.key() == Qt.Key_R:
            self.control_panel.reset_rotation()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.control_panel.add_point()
        elif event.key() == Qt.Key_Escape:
            self.close()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
