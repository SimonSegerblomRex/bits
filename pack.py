import numpy as np


def pack(x, count):
    bits = np.unpackbits(
        x.astype("<u2", copy=False).view("u1").reshape(-1, 2),
        axis=1,
        bitorder="little",
        count=count,
    )
    return np.packbits(bits, bitorder="little")


def unpack(buf, count):
    bits = np.unpackbits(buf, bitorder="little")
    bits = bits.reshape((-1, count))
    bytes_ = np.packbits(bits, axis=1, bitorder="little").reshape(-1)
    if count > 8:
        return bytes_.view("<u2")
    return bytes_.astype(np.uint16)


for bitdepth in range(1, 17):
    width, height = 1920, 1088
    rng = np.random.default_rng()
    image = rng.integers(1 << bitdepth, size=(height, width), dtype=np.uint16)

    encoded = pack(image, count=bitdepth)
    print(f"Compression ratio: {image.nbytes / encoded.nbytes:.3}")
    decoded = unpack(encoded, count=bitdepth).reshape((height, width))
    np.testing.assert_array_equal(image, decoded)
