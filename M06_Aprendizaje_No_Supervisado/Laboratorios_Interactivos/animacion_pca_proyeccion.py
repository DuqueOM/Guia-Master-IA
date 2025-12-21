import numpy as np
from manim import (
    BLUE_E,
    DOWN,
    RED,
    UP,
    YELLOW,
    Create,
    Dot,
    FadeIn,
    Line,
    NumberPlane,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
)


class AnimacionPCAProyeccion(Scene):
    def construct(self):
        title = Text("PCA: proyección sobre PC1", font_size=36).to_edge(UP)
        self.add(title)

        plane = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            },
        )
        plane.add_coordinates()
        self.play(Create(plane))

        rng = np.random.default_rng(7)
        Z = rng.normal(size=(140, 2))
        A = np.array([[2.5, 0.9], [0.0, 0.7]])
        X = Z @ A.T
        Xc = X - X.mean(axis=0, keepdims=True)

        cov = (Xc.T @ Xc) / (len(Xc) - 1)
        vals, vecs = np.linalg.eigh(cov)
        u = vecs[:, np.argsort(vals)[::-1][0]]
        u = u / np.linalg.norm(u)

        dots = VGroup(
            *[Dot(plane.c2p(float(x), float(y)), radius=0.03, color=RED) for x, y in Xc]
        )
        self.play(FadeIn(dots))

        pc1 = Line(
            plane.c2p(float(-5 * u[0]), float(-5 * u[1])),
            plane.c2p(float(5 * u[0]), float(5 * u[1])),
            color=YELLOW,
        )
        label = Text("PC1", font_size=28).next_to(pc1, DOWN)
        self.play(Create(pc1), Write(label))

        proj_dots = VGroup()
        for x, y in Xc:
            p = np.array([x, y], dtype=float)
            t = float(p @ u)
            p_hat = t * u
            proj_dots.add(
                Dot(plane.c2p(float(p_hat[0]), float(p_hat[1])), radius=0.03, color=RED)
            )

        self.play(Transform(dots, proj_dots), run_time=2)
        self.wait(2)
