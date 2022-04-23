import numpy as np


def pack(x, count):
    dtype_bytes = x.dtype.itemsize
    bits = np.unpackbits(
        x.astype(f"<u{dtype_bytes}", copy=False).view("u1").reshape(-1, dtype_bytes),
        axis=1,
        bitorder="little",
        count=count,
    )
    return np.packbits(bits, bitorder="little")


def unpack(buf, count):
    bits = np.unpackbits(buf, bitorder="little").reshape((-1, count))
    bytes_ = np.packbits(bits, axis=1, bitorder="little")
    itemsize = np.min_scalar_type((1 << count) -1).itemsize
    if bytes_.shape[1] < itemsize:
        bytes_ = np.pad(bytes_, ((0, 0), (0, itemsize - bytes_.shape[1])))
    return bytes_.reshape(-1).view(f"<u{itemsize}")


for bitdepth in range(1, 65):
    width, height = 1920, 1088
    rng = np.random.default_rng()
    itemsize = np.min_scalar_type((1 << bitdepth) -1).itemsize
    image = rng.integers(1 << bitdepth, size=(height, width), dtype=f"<u{itemsize}")

    encoded = pack(image, count=bitdepth)
    print(f"Compression ratio: {image.nbytes / encoded.nbytes:5.2f} (bitdepth: {bitdepth:2})")
    decoded = unpack(encoded, count=bitdepth).reshape((height, width))
    np.testing.assert_array_equal(image, decoded)
