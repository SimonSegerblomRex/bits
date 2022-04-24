import numpy as np


def pack(array, bitdepth, bitorder="little"):
    """Pack values with specified bit depth.

    Parameters
    ----------
    array : ndarray
        An array of unsigned integers.
    bitdepth : int
        The number of bits to use for each value.
        Must be large enough to fit the maximum value
        of the input array.
    bitorder: {'big', 'little'}, optional
        The order to pack the bits.
        Defaults to 'little'.

    Returns
    -------
    packed : ndarray
        1-D array of type uint8.
    """
    itemsize = array.dtype.itemsize
    # Convert input array to little endian byte order and use little
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


def decode(buf, bitdepth, bitorder="little"):
    """Decode bytes.

    Parameters
    ----------
    buf : bytearray or uint8 ndarray
        Input buffer with packed bits.
    bitdepth : int
        The number of bits representing a value in
        the input buffer.
    bitorder: {'big', 'little'}
        The order which the bits in the input buffer
        are packed with.
        Defaults to 'little'.

    Returns
    -------
    decoded : ndarray
        Array with decoded values of unsigned type
        large enough to fit values with the specified
        bit depth.
    """
    bits = np.unpackbits(buf, bitorder=bitorder).reshape(-1, bitdepth)
    if bitorder == "big":
        bits = bits[:, ::-1]
    bytes_ = np.packbits(bits, axis=1, bitorder="little")
    itemsize = np.min_scalar_type((1 << bitdepth) - 1).itemsize
    if bytes_.shape[1] < itemsize:
        bytes_ = np.pad(bytes_, ((0, 0), (0, itemsize - bytes_.shape[1])))
    return bytes_.reshape(-1).view(f"<u{itemsize}")


def main():
    """Test various input arrays."""
    for bitorder in ["little", "big"]:
        print(f"Bitorder: {bitorder}")
        for bitdepth in range(1, 65):
            # Generate input
            width, height = 1920, 1088
            rng = np.random.default_rng()
            itemsize = np.min_scalar_type((1 << bitdepth) - 1).itemsize
            image = rng.integers(1 << bitdepth, size=(height, width), dtype=f"u{itemsize}")
            # Pack bits
            encoded = pack(image, bitdepth=bitdepth, bitorder=bitorder)
            print(
                f"Compression ratio: {image.nbytes / encoded.nbytes:5.2f} (bitdepth: {bitdepth:2})"
            )
            # Unpack bits
            decoded = decode(encoded, bitdepth=bitdepth, bitorder=bitorder).reshape(height, width)
            np.testing.assert_array_equal(image, decoded)


if __name__ == "__main__":
    main()
