from utils import write_video_frames


def main():
    from sys import argv
    video_path = argv[1]
    output_dir = argv[2]

    write_video_frames(video_path, output_dir, fps=10, img_size=(224, 224))


if __name__ == '__main__':
    main()
