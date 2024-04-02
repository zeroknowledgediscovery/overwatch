#!/bin/bash


# Check if the models directory exists
if [ ! -d "models" ]; then
    echo "The models directory does not exist."
    exit 1
fi

# Check if models.tgz exists or if there are any [0-9]*model.json files
if [ ! -f "models/models.tgz" ] && [ -z "$(find models -regex 'models/[0-9]*model.json')" ]; then
    echo "Neither models.tgz nor [0-9]*model.json files are present in the models directory."
    exit 1
fi


# Define the target directory for collecting files
target_dir="collected_assets_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$target_dir"

# Check and optionally create models.tgz if it doesn't exist
if [ ! -f "models/models.tgz" ]; then
    # Find and compress [0-9]*model.json files into models.tgz
    find models -regex 'models/[0-9]*model.json' -exec tar -czf "models/models.tgz" -C models {} +
fi

# Check and optionally create modelprop.tgz if it doesn't exist
if [ ! -f "models/modelprop.tgz" ]; then
    tar -czf "models/modelprop.tgz" -C models modelprop.csv
fi



# Copy the specified files and directories into the target directory
cp models/models.tgz "$target_dir/"
cp config.yaml "$target_dir/"
cp models/sim.csv "$target_dir/"
cp models/modelprop.tgz "$target_dir/"

# Create a README.md file with the current date
echo "Collected assets on $(date '+%Y-%m-%d %H:%M:%S')" > "$target_dir/README.md"

# Tar the directory and remove the original
tar -czf "${target_dir}.tar.gz" "$target_dir"
rm -rf "$target_dir"

echo "Assets collected and compressed into ${target_dir}.tar.gz"