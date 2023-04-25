import cv2
import torch
import numpy as np
import argparse
import os

cuda = torch.device('cuda')
parser = argparse.ArgumentParser(prog="waternet",
                                 description="Underwater Image Enhancement")

# torch.dynamo.config.suppress_errors = True
parser.add_argument('filename')
parser.add_argument('codec')
if __name__ == '__main__':
    args = parser.parse_args()
    print("Loading WaterNet from Torch Hub ...")
    preprocess, postprocess, model = torch.hub.load(
        'tnwei/waternet', 'waternet')
    model.eval()
    model.cuda()

    # video_frame = cv2.cuda_GpuMat()

    # model = torch.compile(model)

    # srgan_checkpoint = "./checkpoint_srgan.pth.tar"

    # srgan_ckpt = os.path.abspath(srgan_checkpoint)

    # srgan = torch.load(srgan_ckpt)
    # print(srgan)
    # srgan = srgan['generator']
    # srgan.eval()
    # srgan.cuda()
    camera = cv2.VideoCapture(args.filename)
    ext = args.filename.split('.')[-1]
    fourcc = cv2.VideoWriter_fourcc(*args.codec)

    if not camera.isOpened():
        print("file not found")
        exit()
    fid = 0
    while camera.isOpened():
        ref, frame = camera.read()
        if ref:
            h, w = frame.shape[:2]
            video_frame = torch.tensor(frame)
            # img = cv2.resize(video_frame, np.int32(np.dot([w, h], 0.25)),
            #                  interpolation=cv2.INTER_LANCZOS4)
            # rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (rgb_ten, wb_ten, he_ten, gc_ten) = preprocess(video_frame)
            for name in ["rgb_ten", "wb_ten", "he_ten", "gc_ten"]:
                locals()[name] = (locals()[name].cuda())

            out_ten = model(rgb_ten, wb_ten, he_ten, gc_ten)
            out_img = postprocess(out_ten).squeeze()
            out_img = cv2.resize(out_img, np.int32(
                (w, h)), interpolation=cv2.INTER_LANCZOS4)
            # out_img.to(cuda)
            # sr_img = srgan(out_img)
        else:
            break
        if not "vw" in locals():
            prefix, ext = args.filename.split('.')
            prefix = os.path.abspath(prefix)
            vw_path = "".join([prefix, "_uwe", ".", ext])
            vw = cv2.VideoWriter(
                vw_path, fourcc, 30, (w, h), True)
        vw.write(out_img)
        print("Processed {}".format(fid))
        fid = fid + 1

        cv2.imshow("Raw", frame)
        cv2.imshow("Proccessed", out_img)
        # cv2.imshow("Super Resolution", sr_img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.destroyAllWindows()

    camera.release()
    vw.release()
