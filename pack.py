import bitarray
import numpy as np
from bitarray.util import int2ba


def pack(x, count):
    itemsize = x.dtype.itemsize
    bits = np.unpackbits(
        x.astype(f"<u{itemsize}", copy=False).view("u1").reshape(-1, itemsize),
        axis=1,
        bitorder="little",
        count=count,
    )
    return np.packbits(bits, bitorder="little")


def unpack(buf, count):
    bits = np.unpackbits(buf, bitorder="little").reshape(-1, count)
    bytes_ = np.packbits(bits, axis=1, bitorder="little")
    itemsize = np.min_scalar_type((1 << count) - 1).itemsize
    if bytes_.shape[1] < itemsize:
        bytes_ = np.pad(bytes_, ((0, 0), (0, itemsize - bytes_.shape[1])))
    return bytes_.reshape(-1).view(f"<u{itemsize}")



for bitdepth in range(1, 65):
    width, height = 16, 16#1920, 1088
    rng = np.random.default_rng()
    itemsize = np.min_scalar_type((1 << bitdepth) - 1).itemsize
    image = rng.integers(1 << bitdepth, size=(height, width), dtype=f"<u{itemsize}")

    encoded = pack(image, count=bitdepth)
    print(f"Compression ratio: {image.nbytes / encoded.nbytes:5.2f} (bitdepth: {bitdepth:2})")

    # bitarray module
    ba = bitarray.bitarray(endian="little")
    code = {i: int2ba(i, length=bitdepth, endian="little") for i in range(1 << bitdepth)}
    ba.encode(code, image.ravel())
    baout = ba.tobytes()
    assert baout == encoded.tobytes()

    decoded = unpack(encoded, count=bitdepth).reshape((height, width))
    np.testing.assert_array_equal(image, decoded)


# TODO:
# MSB
# >>> np.unpackbits(np.array([3], dtype="<u2").view("u1"), bitorder="little")
# array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)
# >>> np.unpackbits(np.array([3], dtype=">u2").view("u1"), bitorder="big")
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=uint8)
