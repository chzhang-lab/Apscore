import os
import concurrent.futures
import subprocess

def opt_target_pdb2pqr(input_file, ph=7.0):
    """Add hydrogens to a single protein"""
    try:
        dir_path = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        output_addh_name = os.path.splitext(base_name)[0] + "_addh.pdb"
        output_addh_path = os.path.join(dir_path, output_addh_name)

        # Create temporary file and remove header
        temp_file_path = os.path.join(dir_path, "temp_no_header.pdb")
        with open(input_file, 'r') as infile, open(temp_file_path, 'w') as outfile:
            for line in infile:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    outfile.write(line)

        # Use pdb2pqr to add hydrogens
        addh_command = [
            "pdb2pqr30",
            "--ff=AMBER",
            f"--with-ph={ph}",
            "--keep-chain",
            temp_file_path,
            output_addh_path
        ]
        # subprocess.run(addh_command)
        # Silent output
        subprocess.run(addh_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\n Successfully added hydrogens, file saved at: {output_addh_path}")

        # Delete temporary file
        os.remove(temp_file_path)
        
        return output_addh_path

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return None


def batch_process_proteins(main_dir, max_workers=4):
    """Batch process proteins for hydrogen addition"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for subdir in os.listdir(main_dir):
            subdir_path = os.path.join(main_dir, subdir)
            if os.path.isdir(subdir_path):
                for pdb_file in os.listdir(subdir_path):
                    if pdb_file.endswith("_pocket.pdb"):
                        pdb_file = os.path.join(subdir_path, pdb_file)
                        futures.append(executor.submit(opt_target_pdb2pqr, pdb_file))

        concurrent.futures.wait(futures)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch add hydrogens to protein files using pdb2pqr.")
    parser.add_argument("--input_dir", type=str, required=True,
        help="Directory containing protein subdirectories with *_pocket.pdb files")
    args = parser.parse_args()

    batch_process_proteins(args.input_dir)
