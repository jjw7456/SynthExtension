# plot_specs.py
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def load_wav(path, sr):
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y

def stft_db(y, n_fft, hop, win_length, window, ref_amp):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop,
                            win_length=win_length, window=window))
    # dB: 공통 참조(ref_amp)로 정규화해 비교 가능하게
    S_db = 20.0 * np.log10(np.maximum(S / (ref_amp + 1e-12), 1e-12))
    return S_db

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="name_pred.wav")
    ap.add_argument("--src",  default="name_src.wav")
    ap.add_argument("--tgt",  default="name_tgt.wav")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--window", type=str, default="hann")
    ap.add_argument("--vmin_db", type=float, default=-80.0)
    ap.add_argument("--out", type=str, default=None, help="파일로 저장할 경로(.png 등)")
    args = ap.parse_args()

    # 1) 로드 (같은 sr로 리샘플)
    y_pred = load_wav(args.pred, args.sr)
    y_src  = load_wav(args.src,  args.sr)
    y_tgt  = load_wav(args.tgt,  args.sr)

    # 2) 공통 참조 앰프 (세 신호의 STFT 진폭 글로벌 최대)
    def max_amp(y):
        S = np.abs(librosa.stft(y, n_fft=args.n_fft, hop_length=args.hop,
                                win_length=args.win, window=args.window))
        return S.max() if S.size else 1.0

    ref_amp = max(max_amp(y_pred), max_amp(y_src), max_amp(y_tgt))
    if ref_amp <= 0:
        ref_amp = 1.0

    # 3) dB 스펙트로그램
    S_pred = stft_db(y_pred, args.n_fft, args.hop, args.win, args.window, ref_amp)
    S_src  = stft_db(y_src,  args.n_fft, args.hop, args.win, args.window, ref_amp)
    S_tgt  = stft_db(y_tgt,  args.n_fft, args.hop, args.win, args.window, ref_amp)

    # 4) 플롯 (한 그림에 3개 서브플롯)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    imgs = []
    for ax, S, title in zip(
        axes,
        [S_pred, S_src, S_tgt],
        [f"Pred: {args.pred}", f"Src: {args.src}", f"Tgt: {args.tgt}"]
    ):
        img = librosa.display.specshow(
            S, sr=args.sr, hop_length=args.hop, x_axis="time", y_axis="linear",
            vmin=args.vmin_db, vmax=0.0, cmap="magma", ax=ax
        )
        imgs.append(img)
        ax.set_title(title)
        ax.set_ylabel("Freq (bins)")
    axes[-1].set_xlabel("Time (s)")

    cbar = fig.colorbar(imgs[-1], ax=axes, orientation="vertical", pad=0.01)
    cbar.set_label("dB (ref = global max)")

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()
