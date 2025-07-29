import torch


def revert_delay_pattern(data, start_idx=0):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    assert data.shape[1] - data.shape[0] >= start_idx
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i + start_idx : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


if __name__ == "__main__":
    # Example usage
    data = torch.tensor(
        [
            [1024, 800, 127, 578, 62, 116, 921, 116, 563, 478],
            [1024, 1024, 873, 217, 434, 741, 848, 848, 341, 429],
            [1024, 1024, 1024, 52, 977, 243, 726, 80, 285, 889],
            [1024, 1024, 1024, 1024, 310, 21, 242, 234, 762, 980],
            [1024, 1024, 1024, 1024, 1024, 700, 700, 510, 68, 794],
            [1024, 1024, 1024, 1024, 1024, 1024, 325, 75, 659, 189],
            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 725, 986, 258],
            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 419, 409],
        ]
    )
    except_data = torch.tensor(
        [
            [1024, 800, 127],
            [1024, 873, 217],
            [1024, 52, 977],
            [1024, 310, 21],
            [1024, 700, 700],
            [1024, 325, 75],
            [1024, 725, 986],
            [1024, 419, 409],
        ]
    )
    print(f"data.shape: {data.shape}")
    recovered_data = revert_delay_pattern(data, start_idx=0)
    assert torch.equal(recovered_data, except_data), (
        f"Reverted data does not match expected data {recovered_data=} {except_data=}"
    )

    data = torch.tensor(
        [
            [1024, 800, 127, 578, 62, 116, 921, 116, 563, 478, 538, 538],
            [1024, 1024, 873, 217, 434, 741, 848, 848, 341, 429, 956, 360],
            [1024, 1024, 1024, 52, 977, 243, 726, 80, 285, 889, 703, 872],
            [1024, 1024, 1024, 1024, 310, 21, 242, 234, 762, 980, 268, 801],
            [1024, 1024, 1024, 1024, 1024, 700, 700, 510, 68, 794, 988, 525],
            [1024, 1024, 1024, 1024, 1024, 1024, 325, 75, 659, 189, 523, 980],
            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 725, 986, 258, 823, 87],
            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 419, 409, 248, 548],
        ]
    )
    except_data = torch.tensor(
        [
            [578, 62],
            [434, 741],
            [243, 726],
            [242, 234],
            [510, 68],
            [659, 189],
            [258, 823],
            [248, 548],
        ]
    )
    print(f"data.shape: {data.shape}")
    recovered_data = revert_delay_pattern(data, start_idx=3)
    assert torch.equal(recovered_data, except_data), (
        f"Reverted data does not match expected data {recovered_data=} {except_data=}"
    )
    print("Revert delay pattern test passed.")

    # end
    data = torch.tensor(
        [
            [1024,  800,  127,  578,   62,  116,  921,  116,  563,  478,  538,  538, 127, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025],
            [1024, 1024,  873,  217,  434,  741,  848,  848,  341,  429,  956,  360, 360,  917, 1025, 1025, 1025, 1025, 1025, 1025, 1025],
            [1024, 1024, 1024,   52,  977,  243,  726,   80,  285,  889,  703,  872, 715,  605,  553, 1025, 1025, 1025, 1025, 1025, 1025],
            [1024, 1024, 1024, 1024,  310,   21,  242,  234,  762,  980,  268,  801, 224,   15,  196,  433, 1025, 1025, 1025, 1025, 1025],
            [1024, 1024, 1024, 1024, 1024,  700,  700,  510,   68,  794,  988,  525, 921,  607,  718,  324,  843, 1025, 1025, 1025, 1025],
            [1024, 1024, 1024, 1024, 1024, 1024,  325,   75,  659,  189,  523,  980, 460,  465,  666,  765,  730,  765, 1025, 1025, 1025],
            [1024, 1024, 1024, 1024, 1024, 1024, 1024,  725,  986,  258,  823,   87, 46,  291,  753, 1006,  699,  348,  581, 1025, 1025],
            [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,  419,  409,  248,  548, 137,  510,  625,  844,  620,  390,  981,  962, 1025]
        ]
    )

    except_data = torch.tensor(
        [[ 538,  127, 1025],
        [ 360,  917, 1025],
        [ 605,  553, 1025],
        [ 196,  433, 1025],
        [ 324,  843, 1025],
        [ 730,  765, 1025],
        [ 348,  581, 1025],
        [ 981,  962, 1025]]
    )
    print(f"data.shape: {data.shape}")
    recovered_data = revert_delay_pattern(data, start_idx=11)
    assert torch.equal(recovered_data, except_data), (
        f"Reverted data does not match expected data {recovered_data=} {except_data=}"
    )

    except_data = torch.tensor(
        [
            [1024, 800, 127, 578, 62, 116, 921, 116, 563, 478, 538, 538, 127, 1025],
            [1024, 873, 217, 434, 741, 848, 848, 341, 429, 956, 360, 360, 917, 1025],
            [1024, 52, 977, 243, 726, 80, 285, 889, 703, 872, 715, 605, 553, 1025],
            [1024, 310, 21, 242, 234, 762, 980, 268, 801, 224, 15, 196, 433, 1025],
            [1024, 700, 700, 510, 68, 794, 988, 525, 921, 607, 718, 324, 843, 1025],
            [1024, 325, 75, 659, 189, 523, 980, 460, 465, 666, 765, 730, 765, 1025],
            [1024, 725, 986, 258, 823, 87, 46, 291, 753, 1006, 699, 348, 581, 1025],
            [1024, 419, 409, 248, 548, 137, 510, 625, 844, 620, 390, 981, 962, 1025],
        ]
    )
    recovered_data = revert_delay_pattern(data)
    assert torch.equal(recovered_data, except_data), (
        f"Reverted data does not match expected data {recovered_data=} {except_data=}"
    )
    print("Revert delay pattern test passed.")

    clip_data = recovered_data.clip(0, 1023)
    final_data = clip_data[:, 1:-1]
    print(final_data)