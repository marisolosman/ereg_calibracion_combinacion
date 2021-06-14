
import os
import sys
import hashlib


def calculate_hash_of_file(file_name):
    # Open,close, read file and calculate MD5 on its contents 
    with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()
    # return hash
    return md5_returned
    

# ==================================================================================================
if __name__ == "__main__":
  
     
    input_file = os.path.join(sys.path[0], "input_files_used")
    output_file = os.path.join(sys.path[0], f"{input_file}_with_hashes")

    # open both files
    with open(input_file, 'r') as firstfile, open(output_file, 'w') as secondfile:
        # read content from first file
        for line in firstfile:
            # remove new line and strip
            file_name = line.replace('file: ', '').replace('\n', '').strip()
            # compute hash
            file_hash = calculate_hash_of_file(file_name)
            # append content to second file
            secondfile.write(f"file: {file_name} hash: {file_hash}\n")
            
