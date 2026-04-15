# Demetria
Demetria is a simple package for `numpy` that implements lazily evaluated two-dimensional scalar and vector fields.

It is named after Demeter, the Greek goddess of agriculture, and by extension probably
also of fields, ploughs, combine harvesters and other agricultural implements.

The fields are stored as functions of $x$ and $y$ and are evaluated lazily at any
point -- over regular grids or irregular meshes (both are vectorized with `numpy`).

### Scalar fields
Fields are defined

    f = ScalarField(lambda x, y: x**2 - y**3)
    g = ScalarField(lambda x, y, x**3 - y)
    h = (f + 3 * g)**2

    xs = np.linspace(-10, 10, 101)
    ys = np.linspace(-10, 10, 101)
    xx, yy = np.meshgrid(xs, ys)
    h(xx, yy)

### Vector fields

    a = VectorField(lambda x, y: (x, -y))
    b = VectorField(lambda x, y: (y, 0))
    c = a * b + f * b

Numerical approximations to divergence and rotation can be also computed at all points.

### Zernike polynomials
Implementation of scalar and vector Zernike polynomials in general form is provided too.
Noll, ANSI and (n, l) indexing are supported for scalar polynomials
and (n, l, r) indexing is used for vector polynomials.
