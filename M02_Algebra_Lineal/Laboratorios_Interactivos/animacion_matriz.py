from manim import (
    BLUE_E,
    DOWN,
    GREEN,
    ORIGIN,
    RED,
    RIGHT,
    UL,
    UP,
    ApplyMatrix,
    Arrow,
    Create,
    FadeIn,
    Matrix,
    NumberPlane,
    Scene,
    Tex,
    Text,
    VGroup,
    Write,
)


class AnimacionMatriz(Scene):
    def construct(self):
        A = [[1, 1], [0, 1]]

        title = Text("Visualizando una Transformación Lineal", font_size=36).to_edge(UP)
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

        e1 = Arrow(ORIGIN, RIGHT, buff=0, color=RED)
        e2 = Arrow(ORIGIN, UP, buff=0, color=GREEN)
        basis = VGroup(e1, e2)

        matrix_mob = Matrix(A).to_corner(UL)

        self.play(Create(plane), Create(basis), FadeIn(matrix_mob))
        self.wait(0.5)

        self.play(ApplyMatrix(A, plane), ApplyMatrix(A, basis), run_time=2)
        self.wait(0.5)

        text = Tex("El espacio se deforma, pero el origen se mantiene.").to_edge(DOWN)
        self.play(Write(text))
        self.wait(2)
