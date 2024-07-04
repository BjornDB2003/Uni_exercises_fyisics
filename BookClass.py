# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 3/04/2024
# .............................................................................
# Imports
# .............................................................................
import glob
from pathlib import Path
from natsort import natsorted

# .............................................................................
# Class implementation
# .............................................................................


class Book:
    """
    A class representing a book from the Gutenberg Project. This class
    processes the book, by sorting the chapters from the file structure given
    by the folder BoekHoofdstukken vand merging them into one text file. This
    class also contains methods to remove the biggest and largest chapter and
    to print information about the book (title, author and file size).

    Attributes:
        - book_id must be a string that contains the book number, this string
          may only contain digits.

    Raises:
        - ValueError: If the book_id string contains non-numerical characters

    Note:
        This class will only work if the BoekHoofdstukken folder is in the same
        working directory and that the folders that the books are sorted and
        merged into are present: Boeken and Volledige_Boeken
    """
    def __init__(self, book_id):
        """
        Initiates the book object with the provided book ID

        :param book_id:
            - book_id must be a string that contains the book number, this
              string may only contain digits.
        Raises:
           - ValueError: If the book_id string contains non-numerical
             characters
        """
        # Checking if book_id only contains digits.
        if book_id.isdigit():
            self._book_id = book_id
        else:
            raise ValueError("book_id must contain only numeric characters.")

    @property
    def book_id(self):
        """
        Getter method for the book_id

        :return: book_id
        """
        return self._book_id

    @book_id.setter
    def book_id(self, book_id):
        """
        Setter method for the book_id

        :param book_id:
            - book_id must be a string that contains the book number, this
              string may only contain digits.

        Raises:
           - ValueError: If the book_id string contains non-numerical
             characters

        :return: book_id
        """
        # Checking if book_id only contains digits.
        if book_id.isdigit():
            self._book_id = book_id
        else:
            raise ValueError("book_id must contain only numeric characters.")

    def process(self):
        """
        Method for processing the books, this includes sorting the chapters,
        merging them into one text file, printing information about the book
        and deleting the biggest and largest chapters.
        """
        self.sort_chapters()
        self.merge_chapters()
        self.info_book()
        self.delete_chapters()

    def sort_chapters(self):
        """
        Method for sorting the chapters of the book into a folder inside the
        Boeken folder.

        Note:
            This method only works when the BoekHoofdstukken and Boeken folders
            are in the current working directory.

        Raises:
           - FIleNotFoundError: If the BoekHoofdstukken folder or the Boeken
             folder are not in the current directory.
        """
        cwd = Path.cwd()
        # Checking if the folders BoekHoofdstukken and Boeken exist.
        if (
            not (Path(cwd / "BoekHoofdstukken").exists()) or
            not (Path(cwd / "Boeken").exists())
        ):
            raise FileNotFoundError("Make sure the folders BoekHoofdstukken "
                                    "and Boeken exist.")
        # Making a folder for the book in the Boeken folder.
        Path(f'{cwd}/Boeken/B{self._book_id}').mkdir()
        # Search for all the chapters in the BoekHoofdstukken folder.
        for chapter in natsorted(glob.iglob(
                f'{cwd}/**/p{self._book_id}.txt', recursive=True)):
            # Manipulating the string, so we just get the chapter number.
            chapter_number = str(chapter) \
                .replace(f'{cwd}/BoekHoofdstukken/', "") \
                .replace(f'p{self._book_id}.txt', "") \
                .replace("/", "")
            # Copying the chapters to the correct folder in Boeken.
            copy_from = Path(chapter)
            copy_to = Path(f"{cwd}/Boeken/B{self._book_id}/" 
                           f"B{self._book_id}-C{chapter_number}.txt")
            copy_to.write_bytes(copy_from.read_bytes())

    def merge_chapters(self):
        """
        Method for merging the chapters of the book into the Volledige_Boeken
        folder.

        Note:
            This method only works when the Boeken and Volledige_Boeken folders
            are in the current working directory.

        Raises:
           - FIleNotFoundError: If the Boeken folder or the Volledige_Boeken
             folder are not in the current directory.
        """
        cwd = Path.cwd()
        # Checking if the folders BoekHoofdstukken and Volledige_Boeken exist.
        if (
            not (Path(cwd / "Boeken").exists()) or
            not (Path(cwd / "Volledige_Boeken").exists())
        ):
            raise FileNotFoundError("Make sure the folders "
                                    "Volledige_Boeken and Boeken exist.")
        # Creating an empty file to append all the chapters to in the
        # Volledige_Boeken folder
        with open(Path(f'{cwd}/Volledige_Boeken/B{self._book_id}.txt'),
                  'w') as fp:
            pass
        # Looping over all the chapters in the folder for this book.
        for chapter in natsorted(glob.iglob(
            f'{cwd}/Boeken/B{self._book_id}/*.txt', recursive=True)):
            # Opening the chapter files, so we can append them with pathlib.
            full_book = open(Path(
                f'{cwd}/Volledige_Boeken/B{self._book_id}.txt'), 'a+')
            chapter = open(chapter, 'r')
            # Appending the chapters to the full book, in order!
            full_book.write(chapter.read())
            # Closing the chapter files again.
            chapter.close()
            full_book.close()

    def info_book(self):
        """
        Method for printing information about the book, this includes:
            - Title
            - Author(s)
            - File size

        Note:
            This method only works when the Volledgie_Boeken folders is in the
            same working directory.

        Raises:
           - FIleNotFoundError: If the Volledgie_Boeken folder is not in the
             current directory.
        """
        cwd = Path.cwd()
        # Checking if the folder Volledige_Boeken exist.
        if (
            not (Path(cwd / "Volledige_Boeken").exists())
        ):
            raise FileNotFoundError("Make sure the folder Volledige_Boeken "
                                    "exists.")
        # Opening the complete book text file to get all the lines from it.
        with open(f'{cwd}/Volledige_Boeken/B{self._book_id}.txt',
                  'r') as openbook:
            # Adding all the lines in a list.
            lines = openbook.readlines()
        index_author = []
        index_title = []
        # Checking if a line contains Author: or Title: .
        for i in range(len(lines)):
            if "Author: " in lines[i]:
                index_author.append(i)
                counter = 1
                # Also seeing if there are more non-empty lines after the line
                # containing Author: if multiple authors are given.
                while i + counter < len(lines) and counter != 0:
                    if not lines[i+counter].isspace():
                        index_author.append(i + counter)
                        counter += 1
                    else:
                        counter = 0
            if "Title: " in lines[i]:
                index_title.append(i)
                counter = 1
                # Also seeing if there are more non-empty lines after the line
                # containing Title: if multiple (sub)titles are given.
                while i + counter < len(lines) and counter != 0:
                    if not lines[i+counter].isspace():
                        index_title.append(i + counter)
                        counter += 1
                    else:
                        counter = 0
        # Changing Author: to Authors: if multiple authors are given.
        if len(index_author) > 1:
            lines[index_author[0]] = (lines[index_author[0]][:6] +
                                      's' +
                                      lines[index_author[0]][6:])
        # Getting the size of the file in kilobytes.
        size = round(Path(f'{cwd}/Volledige_Boeken/B{self._book_id}.txt')
                     .stat().st_size/1000, 2)
        # Printing all the book information.
        print("..............................................................."
              ".....")
        print(f"Book: B{self._book_id}")
        for numb in index_title + index_author:
            print(lines[numb][:-1])
        print(f"Filesize: {size} kB")
        print("..............................................................."
              ".....")

    def delete_chapters(self):
        """
        Method for deleting the biggest and smallest chapter of the book.

        Note:
            This method only works when the Boeken folder is in the current
            working directory.

        Raises:
           - FIleNotFoundError: If the Boeken folder is not in the current
            working directory.
        """
        cwd = Path.cwd()
        # Checking if the folder Boeken exists.
        if (
            not (Path(cwd / "Boeken").exists())
        ):
            raise FileNotFoundError("Make sure the folder Boeken "
                                    "exists.")
        chapter_dict = {}
        # Looping over all the chapters in the folder for this book.
        for chapter in natsorted(glob.iglob(
            f'{cwd}/Boeken/B{self._book_id}/*.txt', recursive=True)):
            # Adding the chapter directory path and file size in a directory.
            chapter_dict[chapter] = Path(chapter).stat().st_size
        # Sorting the directory according to the file size.
        sorted_chapter_dict = dict(sorted(chapter_dict.items(),
                                          key=lambda x: x[1]))
        # Deleting the smallest and biggest chapters.
        Path(list(sorted_chapter_dict.keys())[0]).unlink()
        Path(list(sorted_chapter_dict.keys())[-1]).unlink()
        # Printing the deleted chapters from the book.
        print("..............................................................."
              ".....")
        print(f"Deleted chapters in Book: B{self._book_id}")
        words = ["smallest chapter:", "biggest chapter:"]
        for i in [0, -1]:
            # Manipulating the string to get the chapter number.
            chapter_name = str(list(sorted_chapter_dict.keys())[i]) \
                .replace(f'{cwd}/Boeken/B{self._book_id}/'
                         f'B{self._book_id}-', "") \
                .replace('.txt', "")
            print(f"{words[i]:<17} {chapter_name:<5} size: "
                  f"{round(sorted_chapter_dict[list(sorted_chapter_dict.keys())[i]]/1000, 2)} kB")
        print("..............................................................."
              ".....")
