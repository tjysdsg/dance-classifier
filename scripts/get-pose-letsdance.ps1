$datadir = "D:\repos\dance-classifier\letsdance\rgb\"
$outdir = "D:\repos\dance-classifier\letsdance-pose\rgb\"

$categories = @(
    # "ballet",
    # "break",
    # "cha",
    # "flamenco",
    # "foxtrot",
    # "jive",
    # "latin",
    # "pasodoble",
    # "quickstep",
    # "rumba",
    # "samba",
    # "square",
    # "swing",
    # "tango",
    # "tap",
    "waltz"
    )
foreach ($cat in $categories) {
    $cat_dir = $datadir + $cat
    $cat_outdir= $outdir + $cat

    New-Item -ItemType Directory -Force -Path $cat_outdir | Out-Null  # mkdir -p

    Write-Output "Processing $cat"

    D:\repos\openpose\bin\OpenPoseDemo.exe --image_dir $cat_dir --write_images $cat_outdir --disable_blending --display 0
}

