$datadir = "D:\repos\dance-classifier\idance\"
$outdir = "D:\repos\dance-classifier\idance-pose\"

$categories = @(
    "ballet",
    "kpop",
    "latin",
    "classical-chinese",
    "modern"
)

foreach ($cat in $categories) {
    $cat_dir = [IO.Path]::Combine($datadir, $cat)

    $videos = Get-ChildItem $cat_dir
    foreach ($video in $videos) {

        $video_path = [IO.Path]::Combine($cat_dir, $video)
        $vid_outdir = [IO.Path]::Combine($outdir, $cat, $video)

        # skip if output already exists
        if (Test-Path -Path $vid_outdir) {
            Write-Output "Skipping $video_path, since $vid_outdir already exists"
            continue
        }

        New-Item -ItemType Directory -Force -Path $vid_outdir | Out-Null  # mkdir -p
        Write-Output "Processing $video_path, saving to $vid_outdir"

        D:\repos\openpose\bin\OpenPoseDemo.exe --video $video_path --write_json $vid_outdir `
            --keypoint_scale 3 --number_people_max 2 `
            --disable_blending --display 0 --render_pose 0
    }
}
