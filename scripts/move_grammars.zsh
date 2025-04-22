#!/bin/zsh

# set paths
data_dir="data"
grammar_dir="${data_dir}/grammars"
old_dir="${data_dir}/old"

# read keep lists into a single array
keep_dirs=()
for file in "${data_dir}/large_subset.txt" "${data_dir}/small_subset.txt"; do
  while IFS= read -r line; do
    [[ -n "$line" ]] && keep_dirs+=("$line")
  done < "$file"
done

# Print out the number of grammars to keep
echo "Keeping ${#keep_dirs[@]} grammars from ${data_dir}/large_subset.txt and ${data_dir}/small_subset.txt"

# move unlisted directories to old/
for dir_path in "${grammar_dir}"/*(/); do
  dir_name="${dir_path##*/}"

  # Check if dir_name is in keep_dirs
  if [[ " ${keep_dirs[@]} " =~ " $dir_name " ]]; then
    echo "Keeping $dir_name"
  else
    echo "Moving $dir_name to $old_dir/"
    mv "$dir_path" "$old_dir/"
  fi
done
