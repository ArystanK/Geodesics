import sys

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QOpenGLWidget,
                             QVBoxLayout, QWidget, QLabel, QLineEdit,
                             QPushButton, QHBoxLayout, QMessageBox,
                             QGroupBox, QGridLayout)


class GeodesicSphere(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation_x = 0
        self.rotation_y = 0
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
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

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

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton and self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()

            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5

            self.last_pos = event.pos()
            self.update()

    def add_point_from_coordinates(self, x, y, z):
        # Normalize the point to lie on the sphere surface
        vector = np.array([x, y, z])
        length = np.linalg.norm(vector)

        if length == 0:
            return False, "Cannot add point at origin (0,0,0)"

        # Normalize to unit sphere
        normalized = vector / length

        # Scale by sphere radius
        point = normalized * self.sphere_radius
        self.points.append(tuple(point))

        # If we have at least 2 points, create geodesics between consecutive points
        if len(self.points) >= 2:
            p1 = self.points[-2]
            p2 = self.points[-1]
            geodesic = self.compute_geodesic(p1, p2)
            self.geodesics.append(geodesic)

        self.update()
        return True, f"Point added at ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"

    def compute_geodesic(self, p1, p2, num_points=100):
        # Normalize points
        p1 = np.array(p1) / np.linalg.norm(p1)
        p2 = np.array(p2) / np.linalg.norm(p2)

        # Compute angle between points
        dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
        angle = np.arccos(dot)

        # Handle antipodal points (opposite sides of sphere)
        # For antipodal points, there are infinite great circles
        # We choose one by rotating around an arbitrary axis
        if abs(dot + 1.0) < 1e-10:  # Antipodal points (dot ≈ -1)
            # Find an arbitrary perpendicular axis
            if abs(p1[0]) > 1e-10 or abs(p1[1]) > 1e-10:
                # Use cross product with z-axis
                axis = np.cross(p1, [0, 0, 1])
            else:
                # If p1 is aligned with z-axis, use cross product with x-axis
                axis = np.cross(p1, [1, 0, 0])
            axis = axis / np.linalg.norm(axis)

            # Generate points by rotating p1 around the axis
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

        # Generate points along the geodesic using spherical interpolation
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

            # Only normalize if the vector has non-zero length
            norm = np.linalg.norm(point)
            if norm > 1e-10:
                point = point / norm * self.sphere_radius
            else:
                # Fallback: use a small epsilon to avoid division by zero
                point = np.array([1e-10, 0, 0]) * self.sphere_radius

            geodesic.append(point)

        return geodesic

    def clear_all_points(self):
        self.points.clear()
        self.geodesics.clear()
        self.update()


def parse_coordinates(text):
    text = text.strip().replace('(', '').replace(')', '')

    parts = text.split(',')

    if len(parts) != 3:
        return None, "Invalid format. Use: x,y,z"

    try:
        x = float(parts[0].strip())
        y = float(parts[1].strip())
        z = float(parts[2].strip())
        return (x, y, z), None
    except ValueError:
        return None, "Invalid numbers. Use numeric values"


class ControlPanel(QWidget):
    def __init__(self, gl_widget, parent=None):
        super().__init__(parent)
        self.add_button = QPushButton("Add Point")
        self.coord_input = QLineEdit()
        self.clear_button = QPushButton("Clear All Points")
        self.status_label = QLabel("Ready")
        self.gl_widget = gl_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Instructions group (compact)
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()

        instructions = QLabel(
            "Right-click and drag to rotate sphere\n"
            "Enter coordinates as x,y,z\n"
            "Example: 0,0,1 for North Pole"
        )
        instructions.setWordWrap(True)
        instructions.setMaximumWidth(250)
        instructions_layout.addWidget(instructions)
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)

        # Coordinate input group
        input_group = QGroupBox("Add Point")
        input_layout = QVBoxLayout()

        # Coordinate input
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Coords:"))

        self.coord_input.setPlaceholderText("x,y,z")
        self.coord_input.setText("0,0,1")
        self.coord_input.setMaximumWidth(150)
        coord_layout.addWidget(self.coord_input)
        input_layout.addLayout(coord_layout)

        # Add button
        self.add_button.clicked.connect(self.add_point)
        input_layout.addWidget(self.add_button)

        # Status label
        self.status_label.setMaximumHeight(40)
        input_layout.addWidget(self.status_label)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Preset buttons group
        preset_group = QGroupBox("Preset Points")
        preset_layout = QGridLayout()

        presets = [
            ("North Pole", "0,0,1"),
            ("South Pole", "0,0,-1"),
            ("Equator X+", "1,0,0"),
            ("Equator X-", "-1,0,0"),
            ("Equator Y+", "0,1,0"),
            ("Equator Y-", "0,-1,0"),
            ("Front", "0,0,1"),
            ("Back", "0,0,-1")
        ]

        for i, (preset_name, coords) in enumerate(presets):
            btn = QPushButton(preset_name)
            btn.setMaximumWidth(100)
            btn.clicked.connect(lambda checked, c=coords: self.set_preset_coords(c))
            row = i // 2
            col = i % 2
            preset_layout.addWidget(btn, row, col)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        self.clear_button.clicked.connect(self.clear_points)
        layout.addWidget(self.clear_button)

        # Add stretch to push everything to top
        layout.addStretch(1)

    def set_preset_coords(self, coords):
        self.coord_input.setText(coords)

    def add_point(self):
        text = self.coord_input.text()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter coordinates")
            return

        coords, error_data = parse_coordinates(text)
        if error_data:
            QMessageBox.warning(self, "Input Error", error_data)
            return

        x, y, z = coords
        success, message = self.gl_widget.add_point_from_coordinates(x, y, z)

        if success:
            self.status_label.setText(message)
        else:
            QMessageBox.warning(self, "Error", message)

    def clear_points(self):
        self.gl_widget.clear_all_points()
        self.status_label.setText("All points cleared")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geodesic Sphere Visualizer")
        self.setGeometry(100, 100, 900, 600)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create OpenGL widget
        self.gl_widget = GeodesicSphere()

        # Create control panel
        self.control_panel = ControlPanel(self.gl_widget)
        self.control_panel.setMaximumWidth(300)

        # Add widgets to main layout
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.gl_widget, 1)  # Give sphere most of the space

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.gl_widget.clear_all_points()
            self.control_panel.status_label.setText("All points cleared")
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