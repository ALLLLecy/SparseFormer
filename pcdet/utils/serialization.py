# serialization utils.
# include: z-order serialization, hilbert serialization, FlattenWindows serialization

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class ZOrderSerialization(nn.Module):
    def __init__(
            self,
            window_shape,
            depth=8
    ):
        super(ZOrderSerialization, self).__init__()
        self.window_shape = window_shape
        self.depth = depth
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                    key
                    | ((x & mask) << (2 * i + 2))
                    | ((y & mask) << (2 * i + 1))
                    | ((z & mask) << (2 * i + 0))
            )
        return key

    def key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z

    def forward(
            self,
            coords,
            sparse_shape,
            shifts,
            **kwargs
    ):
        """
        Args:
            coords: [L, 3: bs, y, x]
        Returns:
        """
        _, ndim = coords.shape
        coords = coords.long()

        EX, EY, EZ = self.encode_lut(coords.device)
        if ndim == 4:
            sparse_shape = sparse_shape[::-1]
            window_shape = self.window_shape
        else:
            z = torch.zeros_like(coords[:, :1])  # shape: [L, 1]
            coords = torch.cat([coords[:, :1], z, coords[:, 1:]], dim=1)
            sparse_shape = sparse_shape[::-1] + [1]
            window_shape = self.window_shape + [1]

        depth = self.depth
        W, H, D = sparse_shape
        win_shape_x, win_shape_y, win_shape_z = window_shape

        if shifts:
            shift_x = math.ceil(win_shape_x / 2)
            shift_y = math.ceil(win_shape_y / 2)
            shift_z = math.ceil(win_shape_z / 2)
        else:
            shift_x, shift_y, shift_z = 0, 0, 0

        bs = coords[..., 0]
        x = (coords[..., 3] + shift_x) % W
        y = (coords[..., 2] + shift_y) % H
        z = (coords[..., 1] + shift_z) % D

        mask = 255 if depth > 8 else (1 << depth) - 1
        key = EX[x & mask] | EY[y & mask] | EZ[z & mask]

        if depth > 8:
            mask = (1 << (depth - 8)) - 1
            key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
            key = key16 << 24 | key

        key = key | (bs.to(key.dtype) << (depth * 3))

        coords2curve = torch.argsort(key)

        return coords2curve


class HilbertSerialization(nn.Module):
    def __init__(
            self,
            window_shape,
            depth=8
    ):
        super(HilbertSerialization, self).__init__()
        self.window_shape = window_shape
        self.depth = depth

    def forward(
            self,
            coords,
            sparse_shape,
            shifts,
            **kwargs
    ):
        """
        Args:
            coords: [l1 + l2 + l3, 3 or 2], order: x, y, [z]
            sparse_shape: [180, 180]
            shifts: true or false
        Returns:
            coords2curve: [n]
            curve2coords: [bs, N]
            embed coords: [bs, N, 9] order: x, y, z, winx, winy, winz, win_in_x, win_in_y, win_in_z
        """
        _, ndim = coords.shape
        coords = coords.long()

        if ndim == 4:
            sparse_shape = sparse_shape[::-1]
            window_shape = self.window_shape
        else:
            z = torch.zeros_like(coords[:, :1])  # shape: [L, 1]
            coords = torch.cat([coords[:, :1], z, coords[:, 1:]], dim=1)
            sparse_shape = sparse_shape[::-1] + [1]
            window_shape = self.window_shape + [1]

        depth = self.depth
        W, H, D = sparse_shape
        win_shape_x, win_shape_y, win_shape_z = window_shape
        if shifts:
            shift_x = math.ceil(win_shape_x / 2)
            shift_y = math.ceil(win_shape_y / 2)
            shift_z = math.ceil(win_shape_z / 2)
        else:
            shift_x, shift_y, shift_z = 0, 0, 0

        bs = coords[..., 0]
        x = (coords[..., 3] + shift_x) % W
        y = (coords[..., 2] + shift_y) % H
        z = (coords[..., 1] + shift_z) % D

        coord_shifts = torch.stack((x, y, z), dim=-1).view(-1, 3)
        code = self.encode(coord_shifts, 3, depth)

        code = code | (bs.to(code.dtype) << (depth * 3))

        coords2curve = torch.argsort(code)

        return coords2curve

    def right_shift(self, binary, k=1, axis=-1):
        """Right shift an array of binary values.

        Parameters:
        -----------
         binary: An ndarray of binary values.

         k: The number of bits to shift. Default 1.

         axis: The axis along which to shift.  Default -1.

        Returns:
        --------
         Returns an ndarray with zero prepended and the ends truncated, along
         whatever axis was specified."""

        # If we're shifting the whole thing, just return zeros.
        if binary.shape[axis] <= k:
            return torch.zeros_like(binary)

        # Determine the padding pattern.
        # padding = [(0,0)] * len(binary.shape)
        # padding[axis] = (k,0)

        # Determine the slicing pattern to eliminate just the last one.
        slicing = [slice(None)] * len(binary.shape)
        slicing[axis] = slice(None, -k)
        shifted = torch.nn.functional.pad(
            binary[tuple(slicing)], (k, 0), mode="constant", value=0
        )

        return shifted

    def binary2gray(self, binary, axis=-1):
        """Convert an array of binary values into Gray codes.

        This uses the classic X ^ (X >> 1) trick to compute the Gray code.

        Parameters:
        -----------
         binary: An ndarray of binary values.

         axis: The axis along which to compute the gray code. Default=-1.

        Returns:
        --------
         Returns an ndarray of Gray codes.
        """
        shifted = self.right_shift(binary, axis=axis)

        # Do the X ^ (X >> 1) trick.
        gray = torch.logical_xor(binary, shifted)

        return gray

    def gray2binary(self, gray, axis=-1):
        """Convert an array of Gray codes back into binary values.

        Parameters:
        -----------
         gray: An ndarray of gray codes.

         axis: The axis along which to perform Gray decoding. Default=-1.

        Returns:
        --------
         Returns an ndarray of binary values.
        """

        # Loop the log2(bits) number of times necessary, with shift and xor.
        shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
        while shift > 0:
            gray = torch.logical_xor(gray, self.right_shift(gray, shift))
            shift = torch.div(shift, 2, rounding_mode="floor")
        return gray

    def encode(self, locs, num_dims, num_bits):
        """Decode an array of locations in a hypercube into a Hilbert integer.

        This is a vectorized-ish version of the Hilbert curve implementation by John
        Skilling as described in:

        Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
          Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

        Params:
        -------
         locs - An ndarray of locations in a hypercube of num_dims dimensions, in
                which each dimension runs from 0 to 2**num_bits-1.  The shape can
                be arbitrary, as long as the last dimension of the same has size
                num_dims.

         num_dims - The dimensionality of the hypercube. Integer.

         num_bits - The number of bits for each dimension. Integer.

        Returns:
        --------
         The output is an ndarray of uint64 integers with the same shape as the
         input, excluding the last dimension, which needs to be num_dims.
        """

        # Keep around the original shape for later.
        orig_shape = locs.shape
        bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
        bitpack_mask_rev = bitpack_mask.flip(-1)

        if orig_shape[-1] != num_dims:
            raise ValueError(
                """
          The shape of locs was surprising in that the last dimension was of size
          %d, but num_dims=%d.  These need to be equal.
          """
                % (orig_shape[-1], num_dims)
            )

        if num_dims * num_bits > 63:
            raise ValueError(
                """
          num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
          into a int64.  Are you sure you need that many points on your Hilbert
          curve?
          """
                % (num_dims, num_bits, num_dims * num_bits)
            )

        # Treat the location integers as 64-bit unsigned and then split them up into
        # a sequence of uint8s.  Preserve the association by dimension.
        locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

        # Now turn these into bits and truncate to num_bits.
        gray = (
            locs_uint8.unsqueeze(-1)
            .bitwise_and(bitpack_mask_rev)
            .ne(0)
            .byte()
            .flatten(-2, -1)[..., -num_bits:]
        )

        # Run the decoding process the other way.
        # Iterate forwards through the bits.
        for bit in range(0, num_bits):
            # Iterate forwards through the dimensions.
            for dim in range(0, num_dims):
                # Identify which ones have this bit active.
                mask = gray[:, dim, bit]

                # Where this bit is on, invert the 0 dimension for lower bits.
                gray[:, 0, bit + 1:] = torch.logical_xor(
                    gray[:, 0, bit + 1:], mask[:, None]
                )

                # Where the bit is off, exchange the lower bits with the 0 dimension.
                to_flip = torch.logical_and(
                    torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                    torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]),
                )
                gray[:, dim, bit + 1:] = torch.logical_xor(
                    gray[:, dim, bit + 1:], to_flip
                )
                gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)

        # Now flatten out.
        gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

        # Convert Gray back to binary.
        hh_bin = self.gray2binary(gray)

        # Pad back out to 64 bits.
        extra_dims = 64 - num_bits * num_dims
        padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

        # Convert binary values into uint8s.
        hh_uint8 = (
            (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
            .sum(2)
            .squeeze()
            .type(torch.uint8)
        )

        # Convert uint8s into uint64s.
        hh_uint64 = hh_uint8.view(torch.int64).squeeze()

        return hh_uint64

    def decode(self, hilberts, num_dims, num_bits):
        """Decode an array of Hilbert integers into locations in a hypercube.

        This is a vectorized-ish version of the Hilbert curve implementation by John
        Skilling as described in:

        Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
          Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

        Params:
        -------
         hilberts - An ndarray of Hilbert integers.  Must be an integer dtype and
                    cannot have fewer bits than num_dims * num_bits.

         num_dims - The dimensionality of the hypercube. Integer.

         num_bits - The number of bits for each dimension. Integer.

        Returns:
        --------
         The output is an ndarray of unsigned integers with the same shape as hilberts
         but with an additional dimension of size num_dims.
        """

        if num_dims * num_bits > 64:
            raise ValueError(
                """
          num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
          into a uint64.  Are you sure you need that many points on your Hilbert
          curve?
          """
                % (num_dims, num_bits)
            )

        # Handle the case where we got handed a naked integer.
        hilberts = torch.atleast_1d(hilberts)

        # Keep around the shape for later.
        orig_shape = hilberts.shape
        bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)
        bitpack_mask_rev = bitpack_mask.flip(-1)

        # Treat each of the hilberts as a s equence of eight uint8.
        # This treats all of the inputs as uint64 and makes things uniform.
        hh_uint8 = (
            hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
        )

        # Turn these lists of uints into lists of bits and then truncate to the size
        # we actually need for using Skilling's procedure.
        hh_bits = (
            hh_uint8.unsqueeze(-1)
            .bitwise_and(bitpack_mask_rev)
            .ne(0)
            .byte()
            .flatten(-2, -1)[:, -num_dims * num_bits:]
        )

        # Take the sequence of bits and Gray-code it.
        gray = self.binary2gray(hh_bits)

        # There has got to be a better way to do this.
        # I could index them differently, but the eventual packbits likes it this way.
        gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

        # Iterate backwards through the bits.
        for bit in range(num_bits - 1, -1, -1):
            # Iterate backwards through the dimensions.
            for dim in range(num_dims - 1, -1, -1):
                # Identify which ones have this bit active.
                mask = gray[:, dim, bit]

                # Where this bit is on, invert the 0 dimension for lower bits.
                gray[:, 0, bit + 1:] = torch.logical_xor(
                    gray[:, 0, bit + 1:], mask[:, None]
                )

                # Where the bit is off, exchange the lower bits with the 0 dimension.
                to_flip = torch.logical_and(
                    torch.logical_not(mask[:, None]),
                    torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]),
                )
                gray[:, dim, bit + 1:] = torch.logical_xor(
                    gray[:, dim, bit + 1:], to_flip
                )
                gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)

        # Pad back out to 64 bits.
        extra_dims = 64 - num_bits
        padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

        # Now chop these up into blocks of 8.
        locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))

        # Take those blocks and turn them unto uint8s.
        # from IPython import embed; embed()
        locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)

        # Finally, treat these as uint64s.
        flat_locs = locs_uint8.view(torch.int64)

        # Return them in the expected shape.
        return flat_locs.reshape((*orig_shape, num_dims))


class FlattenWindowsSerialization(nn.Module):
    def __init__(
            self,
            window_shape,
            win_version
    ):
        """
        Args:
            window_shape: [lvl_num, 3], order: x, y, z
            win_version: 'v1', 'v2', 'v2e', 'v3', 'v3e'
        """
        super(FlattenWindowsSerialization, self).__init__()
        self.window_shape = window_shape
        self.win_version = win_version

    def forward(
            self,
            coords,
            sparse_shape,
            shifts,
            mapping_name=None,
            **kwargs
    ):
        """
        Args:
            coords: [l1 + l2 + l3, 3 or 2], order: x, y, [z]
            sparse_shape: [180, 180]
            shifts: true or false
            mapping_name: x or y
        Returns:
            coords2curve: [bs, N, 3 or 2], order: x, y, [z]
            curve2coords: [bs, N, 3 or 2], order: x, y, [z]
            embed coords: [bs, N, 9 or 6] order: x, y, [z], winx, winy, [winz], win_in_x, win_in_y, [win_in_z]
            seq coords: [bs, N, 1]
        """
        _, ndim = coords.shape
        coords = coords.long()

        if ndim == 4:
            sparse_shape = sparse_shape[::-1]
            window_shape = self.window_shape
        else:
            z = torch.zeros_like(coords[:, :1])  # shape: [L, 1]
            coords = torch.cat([coords[:, :1], z, coords[:, 1:]], dim=1)
            sparse_shape = sparse_shape[::-1] + [1]
            window_shape = self.window_shape + [1]

        get_win = self.win_version

        if get_win == 'v1':
            # TODO: add embedding coords
            for shifted in [False]:
                (
                    n2,
                    m2,
                    n1,
                    m1,
                    x1,
                    y1,
                    x2,
                    y2,
                ) = self.get_window_coors_shift_v1(coords, sparse_shape, window_shape)
                if mapping_name == 'x':
                    vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
                    vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                    coords2curve = torch.argsort(vx)

                else:
                    vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
                    vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                    coords2curve = torch.argsort(vy)

        elif get_win == 'v2':
            batch_win_inds_x, batch_win_inds_y, coors_in_win, win_coors, coords \
                = self.get_window_coors_shift_v2(coords, sparse_shape, window_shape, shifts)

            if mapping_name == 'x':
                vx = batch_win_inds_x * window_shape[0] * window_shape[1] * window_shape[2]
                vx += coors_in_win[..., 2] * window_shape[1] * window_shape[2] + coors_in_win[..., 1] * \
                      window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vx)
            else:
                vy = batch_win_inds_y * window_shape[0] * window_shape[1] * window_shape[2]
                vy += coors_in_win[..., 1] * window_shape[0] * window_shape[2] + coors_in_win[..., 2] * \
                      window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vy)

        elif get_win == 'v2e':
            batch_win_inds_x, batch_win_inds_y, coors_in_win, win_coors, coords \
                = self.get_window_coors_shift_v2(coords, sparse_shape, window_shape, shifts=shifts)

            vx = batch_win_inds_x * window_shape[0] * window_shape[1] * window_shape[2]
            vy = batch_win_inds_y * window_shape[0] * window_shape[1] * window_shape[2]

            if mapping_name == 'xx':
                vx_xy = vx + coors_in_win[..., 2] * window_shape[1] * window_shape[2] + coors_in_win[..., 1] * \
                        window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vx_xy)

            elif mapping_name == 'xy':
                vx_yx = vx + coors_in_win[..., 1] * window_shape[0] * window_shape[2] + coors_in_win[..., 2] * \
                        window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vx_yx)
            elif mapping_name == "yx":
                vy_xy = vy + coors_in_win[..., 2] * window_shape[1] * window_shape[2] + coors_in_win[..., 1] * \
                        window_shape[2] + coors_in_win[..., 0]

                coords2curve = torch.argsort(vy_xy)
            else:
                vy_yx = vy + coors_in_win[..., 1] * window_shape[0] * window_shape[2] + coors_in_win[..., 2] * \
                        window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vy_yx)

        elif get_win == 'v3':
            batch_win_inds_x, batch_win_inds_y, coors_in_win, win_coors, coords \
                = self.get_window_coors_shift_v3(coords, sparse_shape, window_shape, shifts=shifts)

            if mapping_name == 'x':
                vx = batch_win_inds_x * window_shape[0] * window_shape[1] * window_shape[2]
                vx += coors_in_win[..., 2] * window_shape[1] * window_shape[2] + coors_in_win[..., 1] * \
                      window_shape[2] + coors_in_win[..., 0]

                coords2curve = torch.argsort(vx)
            else:
                vy = batch_win_inds_y * window_shape[0] * window_shape[1] * window_shape[2]
                vy += coors_in_win[..., 1] * window_shape[0] * window_shape[2] + coors_in_win[..., 2] * \
                      window_shape[2] + coors_in_win[..., 0]

                coords2curve = torch.argsort(vy)

        elif get_win == 'v3e':
            batch_win_inds_x, batch_win_inds_y, coors_in_win, win_coors, coords \
                = self.get_window_coors_shift_v3(coords, sparse_shape, window_shape, shifts)

            vx = batch_win_inds_x * window_shape[0] * window_shape[1] * window_shape[2]
            vy = batch_win_inds_y * window_shape[0] * window_shape[1] * window_shape[2]

            if mapping_name == 'xx':
                vx_xy = vx + coors_in_win[..., 2] * window_shape[1] * window_shape[2] + coors_in_win[..., 1] * \
                        window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vx_xy)

            elif mapping_name == 'xy':
                vx_yx = vx + coors_in_win[..., 1] * window_shape[0] * window_shape[2] + coors_in_win[..., 2] * \
                        window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vx_yx)

            elif mapping_name == "yx":
                vy_xy = vy + coors_in_win[..., 2] * window_shape[1] * window_shape[2] + coors_in_win[..., 1] * \
                        window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vy_xy)

            else:
                vy_yx = vy + coors_in_win[..., 1] * window_shape[0] * window_shape[2] + coors_in_win[..., 2] * \
                        window_shape[2] + coors_in_win[..., 0]
                coords2curve = torch.argsort(vy_yx)
        else:
            raise NotImplementedError

        return coords2curve

    @torch.inference_mode()
    def get_window_coors_shift_v3(self, coords, sparse_shape, window_shape, shifts=False):
        sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
        win_shape_x, win_shape_y, win_shape_z = window_shape
        # bs, N, _ = coords.shape

        if shifts:
            shift_x, shift_y, shift_z = math.ceil(win_shape_x / 2), math.ceil(win_shape_y / 2), math.ceil(
                win_shape_z / 2)
        else:
            shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

        max_num_win_x = int(math.ceil(sparse_shape_x / win_shape_x))
        max_num_win_y = int(math.ceil(sparse_shape_y / win_shape_y))
        max_num_win_z = int(math.ceil(sparse_shape_z / win_shape_z))
        max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

        x = (coords[..., 3] + shift_x) % sparse_shape_x  # [bs, N]
        y = (coords[..., 2] + shift_y) % sparse_shape_y  # [bs, N]
        z = (coords[..., 1] + shift_z) % sparse_shape_z  # [bs, N]

        win_coors_x = x // win_shape_x  # [bs, N]
        win_coors_y = y // win_shape_y  # [bs, N]
        win_coors_z = z // win_shape_z  # [bs, N]

        coors_in_win_x = x % win_shape_x  # [bs, N]
        coors_in_win_y = y % win_shape_y  # [bs, N]
        coors_in_win_z = z % win_shape_z  # [bs, N]

        batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                           win_coors_y * max_num_win_z + win_coors_z
        batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                           win_coors_x * max_num_win_z + win_coors_z

        coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)  # [bs, N, 3]
        win_coords = torch.stack([win_coors_z, win_coors_y, win_coors_x], dim=-1)  # [bs, N, 3]
        shifts_coors = torch.stack([z, x, y], dim=-1)

        return batch_win_inds_x, batch_win_inds_y, coors_in_win, win_coords, shifts_coors

    @torch.inference_mode()
    def get_window_coors_shift_v2(self, coords, sparse_shape, window_shape, shifts=False):
        sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
        win_shape_x, win_shape_y, win_shape_z = window_shape
        if shifts:
            shift_x, shift_y, shift_z = math.ceil(win_shape_x / 2), math.ceil(win_shape_y / 2), math.ceil(
                win_shape_z / 2)
        else:
            shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

        max_num_win_x = int(math.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
        max_num_win_y = int(math.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
        max_num_win_z = int(math.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

        max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

        x = coords[..., 3] + shift_x
        y = coords[..., 2] + shift_y
        z = coords[..., 1] + shift_z

        win_coors_x = x // win_shape_x
        win_coors_y = y // win_shape_y
        win_coors_z = z // win_shape_z

        coors_in_win_x = x % win_shape_x
        coors_in_win_y = y % win_shape_y
        coors_in_win_z = z % win_shape_z

        batch_win_inds_x = (
                coords[:, 0] * max_num_win_per_sample +
                win_coors_x * max_num_win_y * max_num_win_z
                + win_coors_y * max_num_win_z
                + win_coors_z
        )  # [bs, N]
        batch_win_inds_y = (
                coords[:, 0] * max_num_win_per_sample +
                win_coors_y * max_num_win_x * max_num_win_z
                + win_coors_x * max_num_win_z
                + win_coors_z
        )  # [bs, N]

        coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
        win_coors = torch.stack([win_coors_z, win_coors_y, win_coors_x], dim=-1)
        shifts_coors = torch.stack([x, y, z], dim=-1)

        return batch_win_inds_x, batch_win_inds_y, coors_in_win, win_coors, shifts_coors

    @torch.inference_mode()
    def get_window_coors_shift_v1(self, coords, sparse_shape, window_shape):
        _, m, n = sparse_shape
        n2, m2, _ = window_shape

        n1 = int(math.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
        m1 = int(math.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

        x = coords[:, 3]
        y = coords[:, 2]

        x1 = x // n2
        y1 = y // m2
        x2 = x % n2
        y2 = y % m2

        return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2
