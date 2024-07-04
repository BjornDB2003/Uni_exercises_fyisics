# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 3/04/2024
# .............................................................................
# Imports
# .............................................................................
import glob
import time
import shutil
import zipfile
from pathlib import Path
from multiprocessing import Pool
from natsort import natsorted
# Import the Book class form BookClass to process the books
from BookClass import Book
from memory_profiler import memory_usage
# .............................................................................
# Defining functions
# .............................................................................


def run(book_id):
    """
    Process a book when given its book ID using the Book class.

    :param book_id:
        - book_id must be a string that contains the book number, this
          string may only contain digits.
    Raises:
        - ValueError: If the book_id string contains non-numerical
          characters
    """
    # checking if book_id only contains numbers.
    if book_id.isdigit():
        # Making the book object and processing it.
        book_to_process = Book(book_id)
        book_to_process.process()
    else:
        raise ValueError("book_id must contain only numeric characters.")


def initiate_folders(cwd_int):
    """
    Creates the necessary folders for processing the books correctly when using
    the Book class and also unzipts the BoekHoofdstukken.zip file. This
    function will check and delete the BoekHoofdstukken, Boeken and
    Volledige_Boeken folders before creating nem ones.

    :param cwd_int:
        - The current working directory where the folders need to be
          created

    Note:
        This function will delete the BoekHoofdstukken, Boeken and
        Volledige_Boeken folder if they are present when running this function
        so make sure there are no valuable files in these folders before
        running this function.
    """
    # Checking if cwd_int is a valid directory path.
    if Path(cwd_int).exists():
        # Deleting folders at the beginning of the program.
        folders_to_delete = [Path(cwd_int / "BoekHoofdstukken"),
                             Path(cwd_int / "Boeken"),
                             Path(cwd_int / "Volledige_Boeken")]
        for i in range(0, 3):
            if folders_to_delete[i].exists() and folders_to_delete[i].is_dir():
                shutil.rmtree(folders_to_delete[i])
                print(f"Removed {folders_to_delete[i]}")
            else:
                print(f"Couldnt remove {folders_to_delete[i]}")

        # Unzipping the BoekHoofdstukken.zip file in the current directory.
        with zipfile.ZipFile('BoekHoofdstukken.zip', 'r') as zip_ref:
            zip_ref.extractall()

        # creating the directories the program will need to process the books.
        path_boeken = Path(cwd_int / "Boeken")
        path_volledige_boeken = Path(cwd_int / "Volledige_Boeken")
        path_boeken.mkdir()
        path_volledige_boeken.mkdir()
    else:
        raise FileNotFoundError("cwd_int must be a valid directory path.")


def clean_up_folders(cwd_cleanup, remove_zip_test):
    """
    Removes the BoekHoofdstukken folder with all the chapters and (optionally)
    also removes the BoekHoofdstukken.zip file.

    :param cwd_cleanup:
        - The current working directory where the folders need to be
          created

    :param remove_zip_test:
        - A Boolean that, when True, makes it so the function also
          removes the BoekHoofdstukken.zip file.
    """
    if not type(remove_zip_test) is bool:
        raise ValueError("remove_zip_test must be a boolean.")
    # Checking if cwd_cleanup is a valid directory path.
    if Path(cwd_cleanup).exists():
        # Deleting the BoekHoofdstukken folder if it's a folder and it exists.
        cwd_cl = cwd_cleanup / "BoekHoofdstukken"
        if Path(f"{cwd_cl}").exists() and Path(f"{cwd_cl}").is_dir():
            shutil.rmtree(Path(f"{cwd_cl}"))
            print(f"Removed {Path(f'{cwd_cl}')}")
        else:
            print(f"Couldnt remove {Path(f'{cwd_cl}')}")

        # Deleting the BoekHoofdstukken.zip file (optional).
        if Path(f"{cwd_cl}.zip").exists() and remove_zip_test:
            Path(f"{cwd_cl}zip").unlink()
            print(f"Removed {Path(f'{cwd_cl}.zip')}")
        else:
            print(f"Didnt remove {Path(f'{cwd_cl}.zip')}")
    else:
        raise FileNotFoundError("cwd_cleanup must be a valid directory path.")

# .............................................................................
# Main functions
# .............................................................................


def main():
    # Optional controls for deleting the BoekHoofdstukken.zp file and to run
    # the processing of the books without multiprocessing.
    remove_zip = False
    run_not_multiprocessing = False
    run_multiprocessing = True

    cwd = Path.cwd()
    # If run_without_multiprocessing = False the program will process the books
    # here without using multiprocessing
    if run_not_multiprocessing:
        # Calling the initiate_folders function to make the necessary folders.
        initiate_folders(cwd)
        # Measure the time it takes to process the books.
        start_time1 = time.time()
        # Looping over every chapter in the chapter folder 0 (this folder
        # contains all the books).
        folder0 = Path(cwd / "BoekHoofdstukken/0")
        for book in natsorted(glob.iglob(f'{folder0}/*.txt',
                                         recursive=True)):
            # Manipulate the Path string and initiate the book object.
            str_to_remove = str(folder0) + "/p"
            book_id_num = str(book).replace(str_to_remove, "")[:-4]
            boek = Book(book_id_num)
            # Process the book.
            boek.process()
        end_time1 = time.time() - start_time1

    if run_multiprocessing:
        # Calling the initiate_folders function to make the necessary folders.
        initiate_folders(cwd)
        # Measure the time it takes to process the books.
        start_time2 = time.time()
        # Looping over every chapter in the chapter folder 0 (this folder
        # contains all the books).
        folder0 = Path(cwd / "BoekHoofdstukken/0")
        book_id_list = []
        for book in natsorted(glob.iglob(f'{folder0}/*.txt',
                                         recursive=True)):
            # Manipulate the Path string and add all the book_id's in a list.
            str_to_remove = str(folder0) + "/p"
            book_id_num = str(book).replace(str_to_remove, "")[:-4]
            book_id_list.append(book_id_num)
        # Initiating the multiprocessing pool.
        p = Pool()
        # Creating a process for every book in the book_id_list.
        p.map(run, book_id_list)
        p.close()
        # Wait for every process to finish.
        p.join()
        end_time2 = time.time() - start_time2

    # Printing the timing results.
    try:
        print(f"Processing all the books without using multiprocessing "
              f"took {end_time1} seconds.")
        print("..............................................................."
              ".........................")
    except NameError:
        pass
    try:
        print(f"Processing all the books using multiprocessing "
              f"took {end_time2} seconds.")
        print("..............................................................."
              "......................... \n")
    except NameError:
        pass

    # Calling the clean_up_folders function to delete the BoekHoofdstukken
    # folder and optionally delete the BoekHoofdstukken.zip file.
    clean_up_folders(cwd, remove_zip)


def profiler(function, *args, **kwargs):
    """
    Function for profiling the main function
    """
    base_memory = memory_usage(max_usage=True)
    memory = memory_usage((function, args, kwargs), max_usage=True)
    memory_used = (memory - base_memory) * 1.048576  # Converting MiB to MB
    print(f'Memory used by {function.__name__}: {round(memory_used, 5)} MB')


profiler(main)
