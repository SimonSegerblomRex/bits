import bitarray
import numpy as np
from bitarray.util import int2ba


def pack(array, bitdepth, bitorder="little"):
    """Pack."""
    itemsize = array.dtype.itemsize
    # Convert input array to little endian byte order and read as little
    # endian bit order to be able to utilize the count parameter.
    bits = np.unpackbits(
        array.astype(f"<u{itemsize}", copy=False).view("u1").reshape(-1, itemsize),
        axis=1,
        bitorder="little",
        count=bitdepth,
    )
    if bitorder == "big":
        bits = bits[:, ::-1]
    return np.packbits(bits, bitorder=bitorder)


def unpack(buf, bitdepth, bitorder="little"):
    """Unpack."""
    bits = np.unpackbits(buf, bitorder=bitorder).reshape(-1, bitdepth)
    if bitorder == "big":
        bits = bits[:, ::-1]
    bytes_ = np.packbits(bits, axis=1, bitorder="little")
    itemsize = np.min_scalar_type((1 << bitdepth) - 1).itemsize
    if bytes_.shape[1] < itemsize:
        bytes_ = np.pad(bytes_, ((0, 0), (0, itemsize - bytes_.shape[1])))
    return bytes_.reshape(-1).view(f"<u{itemsize}")


for bitorder in ["little", "big"]:
    print(f"Bitorder: {bitorder}")
    for bitdepth in range(1, 65):
        width, height = 1920, 1088
        rng = np.random.default_rng()
        itemsize = np.min_scalar_type((1 << bitdepth) - 1).itemsize
        image = rng.integers(1 << bitdepth, size=(height, width), dtype=f"u{itemsize}")

        encoded = pack(image, bitdepth=bitdepth, bitorder=bitorder)
        print(f"Compression ratio: {image.nbytes / encoded.nbytes:5.2f} (bitdepth: {bitdepth:2})")

        if 0:
            # bitarray module
            ba = bitarray.bitarray(endian=bitorder)
            code = {i: int2ba(i, length=bitdepth, endian=bitorder) for i in range(1 << bitdepth)}
            ba.encode(code, image.ravel())
            baout = ba.tobytes()
            assert baout == encoded.tobytes()

        decoded = unpack(encoded, bitdepth=bitdepth, bitorder=bitorder).reshape((height, width))
        np.testing.assert_array_equal(image, decoded)
