import torch


def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    if not self.guide_mask:
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    else:
        len_keep = int(L * (1 - mask_ratio))
        len_l = 96
        len_m = 64
        len_h = 36
        low = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 40, 41, 42, 43, 54, 55, 56, 57, 68, 69, 70, 71, 82, 83, 84, 85, 96, 97, 98, 99, 110, 111, 112, 113,
             124, 125, 126, 127, 138, 139, 140, 141, 152, 153, 154, 155, 166, 167, 168, 169, 170, 171, 172, 173, 174,
             175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
             195]).to(x.device)
        mid = torch.tensor(
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 58, 59, 66, 67, 72, 73, 80,
             81, 86, 87, 94, 95, 100, 101, 108, 109, 114, 115, 122, 123, 128, 129, 136, 137, 142, 143, 144, 145, 146,
             147, 148, 149, 150, 151, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165]).to(x.device)
        hig = torch.tensor(
            [60, 61, 62, 63, 64, 65, 74, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 102, 103, 104, 105, 106, 107, 116,
             117, 118, 119, 120, 121, 130, 131, 132, 133, 134, 135]).to(x.device)
        noise = torch.zeros(N, L, device=x.device)  # noise in [0, 1]
        noise_l = torch.rand(N, len_l, device=x.device) * (1 - (1 - mask_ratio) * 1 / 2) ** 0.5
        noise_m = torch.rand(N, len_m, device=x.device)
        noise_h = torch.rand(N, len_h, device=x.device) * 1 / (1 - (1 - mask_ratio) * 1 / 2)
        noise = noise.scatter_(1, low.unsqueeze(0).repeat(N, 1), noise_l)
        noise = noise.scatter_(1, mid.unsqueeze(0).repeat(N, 1), noise_m)
        noise = noise.scatter_(1, hig.unsqueeze(0).repeat(N, 1), noise_h)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore