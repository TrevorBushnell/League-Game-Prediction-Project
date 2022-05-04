from mysklearn import myutils

# TODO: copy your mypytable.py solution from PA2-PA6 here

from mysklearn import myutils

# TODO: copy your mypytable.py solution from PA2-PA3 here
import copy
import csv
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        N = len(self.data)
        M = len(self.column_names)
        return N, M # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        table = []
        if type(col_identifier) == int:
            for i in range(len(self.data)):
                if (include_missing_values == False) and (self.data[i][col_identifier] =="NA"):
                    pass
                else:
                    table.append(self.data[i][col_identifier])
        else: 
            try:
                index = self.column_names.index(col_identifier)
                for i in range(len(self.data)):
                    if (include_missing_values == False) and (self.data[i][index] =="NA"):
                        pass
                    else:
                        table.append(self.data[i][index])
            except: 
                raise ValueError("Invalid col_identifier")
        return table # TODO: fix this

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j]= float(self.data[i][j])
                    
                except ValueError:
                    self.data[i][j]= self.data[i][j]
        # TODO: fix this

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort()
        for i in range(len(row_indexes_to_drop)):
            if i == 0:
                self.data.pop(row_indexes_to_drop[i])
            else:
                self.data.pop(row_indexes_to_drop[i]-i) 
        pass # TODO: fix this

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table= []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter = ',')
            for row in csv_reader:
                table.append(row)
            self.column_names = table[0]
            table.pop(0)
            self.data = table
            self.convert_to_numeric()
            # TODO: finish this
            return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, 'w',) as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(self.column_names)
            for row in self.data:
                filewriter.writerow(row)
        pass # TODO: fix this

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        index = []
        for name in key_column_names:
            index.append(self.column_names.index(name))
        key_list = []
        table = []
        for i in range(len(self.data)):
            key = get_key(index,self.data[i])
            if (key in key_list):
                table.append(i)
            else:
                key_list.append(key)
        return table # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        i = 0
        while i < len(self.data):
            count = 0
            for j in range(len(self.data[i])):
                if self.data[i][j]=="NA" or self.data[i][j]=="N/A" or self.data[i][j] == '':
                    count +=1
            if count >0:
                self.data.pop(i)
            else: 
                i = i+1
        pass # TODO: fix this

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        for name in col_name:
            index = self.column_names.index(name)
            lst = self.get_column(name)
            total = 0
            num = 0
            for i in range(len(lst)):
                if type(lst[i]) == str:
                    lst[i] = lst[i]
                else:
                    total = total + lst[i]
                    num +=1
            mean = total / num
            for i in range(len(self.data)):
                if self.data[i][index] == "NA" or self.data[i][index]=="N/A" or self.data[i][index] == '':
                    self.data[i][index] = mean
        # TODO: fix this

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        sum_stats = []
        for name in col_names:

            self.convert_to_numeric()
            col = self.get_column(name, include_missing_values=False)
            col.sort()
            if len(col) ==0:
                return MyPyTable()
            min = col[0]
            max = col[-1]
            mid = 0
            avg = 0
            median = 0
            for i in range(len(col)):
                avg = avg + col[i]
            avg = avg/len(col)
            midpoint = len(col)//2
            if (len(col) %2) ==0:
                median = (col[midpoint-1]+col[midpoint])/2
            else:
                median = col[midpoint]
            mid = (max +min)/2
            sum_stats.append([name, min, max, mid, avg, median])
        
        return MyPyTable(column_names = ["attribute", "min", "max", "mid", "avg", "median"],data=sum_stats) # TODO: fix this

    def convert_percent(self,col_name):
        col = self.get_column(col_name)
        for i in range(len(col)):
            value = col[i].split("%")
            cent = value[0]
            col[i] = float(cent)
        index = self.column_names.index(col_name)
        for i in range(len(col)):
            self.data[i][index] = col[i]
    
    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        index1 = []
        for name in key_column_names:
            index1.append(self.column_names.index(name))
        index2 = []
        for name in key_column_names:
            index2.append(other_table.column_names.index(name))
        place_holder_col = other_table.column_names.copy()
        for i in range(len(index2)):
            place_holder_col.pop(index2[i]-i)
        joined_col_names = self.column_names + place_holder_col
        joined_rows = []
        for row1 in self.data:
            for row2 in other_table.data:
                key1 = ""
                key2 = ""
                for value in index1:
                    key1 +=str(row1[value])
                for value in index2:
                    key2 +=str(row2[value])
                if key1 == key2:
                    index2.sort()
                    place_holder_row = row2.copy()
                    for i in range(len(index2)):
                        place_holder_row.pop(index2[i]-i)
                    joined_rows.append(row1+place_holder_row)
                
        return MyPyTable(column_names=joined_col_names, data=joined_rows) # TODO: fix this

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        index1 = []
        for name in key_column_names:
            index1.append(self.column_names.index(name))
        index2 = []
        for name in key_column_names:
            index2.append(other_table.column_names.index(name))
        place_holder_col = other_table.column_names.copy()
        for i in range(len(index2)):
            place_holder_col.pop(index2[i]-i)
        joined_col_names = self.column_names + place_holder_col
        joined_rows = []
        for row1 in self.data:
            paired = False
            for row2 in other_table.data:
                key1 = ""
                key2 = ""
                for value in index1:
                    key1 +=str(row1[value])
                for value in index2:
                    key2 +=str(row2[value])
                if key1 == key2:
                    index2.sort()
                    place_holder_row = row2.copy()
                    for i in range(len(index2)):
                        place_holder_row.pop(index2[i]-i)
                    joined_rows.append(row1+place_holder_row)
                    paired = True
            if paired == False:
                joined_rows.append(row1+(len(other_table.column_names)-len(index2))*["NA"])
        
        for row1 in other_table.data:
            count = 0
            for row2 in self.data:
                key_joined= get_key(index1,row2)
                key2 = get_key(index2,row1)
                if key2 == key_joined:
                    count +=1
            if count ==0:
                index2.sort()
                place_holder_row = row1.copy()
                null_row = len(self.column_names) *["NA"]
                for i in range(len(index2)):
                    null_row[index1[i]] = place_holder_row.pop(index2[i]-i)
                joined_rows.append(null_row + place_holder_row)
                
        return MyPyTable(column_names=joined_col_names, data=joined_rows)# TODO: fix this

def get_key(indexs,row):
        key = ""
        for value in indexs:
            key +=str(row[value])
        return key
