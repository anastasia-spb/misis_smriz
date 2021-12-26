import numpy as np
import cvxpy


class Parameters:
    def __init__(self):
        # 'N' is number of available technologies
        self.N = 12
        # 'M' is number of available products
        self.M = 6
        # 'A' contains information about how many tonnes of each product we can get by using chosen technology and
        # one tonn of resource, e.g. a_{ij} defines how many tonn of product i we will get by using a j technology
        self.A = np.array([[0.000, 0.000, 0.311, 0.311, 0.000, 0.249, 0.000, 0.372, 0.373, 0.000, 0.000, 0.372],
                           [0.265, 0.352, 0.000, 0.000, 0.000, 0.000, 0.211, 0.361, 0.000, 0.265, 0.000, 0.316],
                           [0.000, 0.630, 0.420, 0.210, 0.505, 0.000, 0.000, 0.000, 0.000, 0.000, 0.588, 0.000],
                           [0.136, 0.000, 0.000, 0.000, 0.328, 0.000, 0.492, 0.000, 0.328, 0.000, 0.273, 0.000],
                           [0.577, 0.000, 0.000, 0.462, 0.138, 0.138, 0.277, 0.000, 0.277, 0.231, 0.115, 0.000],
                           [0.000, 0.000, 0.243, 0.000, 0.000, 0.584, 0.000, 0.292, 0.000, 0.486, 0.000, 0.292]])

        # 'eps' defines amount of waste in tonnes we get from one tonn of resource
        self.eps = np.array([0.022, 0.018, 0.026, 0.017, 0.029, 0.029,
                             0.020, 0.020, 0.022, 0.018, 0.024, 0.020]).reshape((1, self.N))

        # 'productivity' defines how much resource in tonnes can be processed by every technology
        self.productivity = np.array([15, 11, 17, 15, 18, 17,
                                      13, 14, 20, 21, 24, 19]).reshape((1, self.N))
        self.inv_productivity = np.reciprocal(self.productivity.astype(np.float32))
        # 'plan' defines how many tonnes of each product we shall produce
        self.plan = np.array([135, 125, 150, 110, 150, 100]).reshape((1, self.M))
        # 'T1' available time for the first machine in hours
        self.T1 = 3
        # 'T2' available time for the second machine in hours
        self.T2 = 6
        # 'N1' contains technologies which could work on the first machine
        self.N1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        # '__N2' contains technologies which could work on the second machine
        self.__N2 = np.array([8, 9, 10, 11, 12])
        # 'N2_ext' contains indices of technologies which could work only on the second machine
        # 'indices_dict' contains mapping between technology in the N1 and its copy in N2_ext
        self.N2_ext, self.indices_dict = self.__split_technologies()
        self.A_ext = self.__extend_matrix(self.A)
        self.A_ext_norm = np.round(self.A_ext / np.sum(self.A_ext, axis=0), 3)
        self.inv_P = self.__extend_matrix(self.inv_productivity)
        self.eps_ext = self.__extend_matrix(self.eps)
        self.N_ext = self.N1.size + self.N2_ext.size

    def __split_technologies(self):
        """ Gets number of technologies which could work on both machines
            and stores them in dictionary, where key is a number of such technologies
            and value it corresponding index of "duplicated technology".
         """
        intersections = set.intersection(set(self.N1), set(self.__N2))
        new_indices = np.arange(self.N + 1, self.N + len(intersections) + 1)
        n2_ext = np.append(list(set(self.__N2) - set(self.N1)), new_indices)
        indices_dict = {list(intersections)[i]: new_indices[i] for i in range(0, len(intersections))}
        return n2_ext, indices_dict

    def __extend_matrix(self, mat):
        """ Duplicates information about technologies which
            could be run on both machines
         """
        if mat.shape[1] != self.N:
            raise ValueError("Matrix has wrong amount of columns!")
        mat_ext = mat.copy()
        for i in self.indices_dict.keys():
            mat_ext = np.c_[mat_ext, mat_ext[:, i - 1]]
        return mat_ext

    def __check_variables_shape(self, x):
        if x.shape != (1, self.N_ext):
            raise ValueError("Variables has wrong shape")

    def get_time_values(self, x, technologies_indices):
        return cvxpy.multiply(x[:, technologies_indices - 1],
                              self.inv_P[:, technologies_indices - 1])

    def get_time_1_values(self, x):
        self.__check_variables_shape(x)
        return self.get_time_values(x, self.N1)

    def get_time_2_values(self, x):
        self.__check_variables_shape(x)
        return self.get_time_values(x, self.N2_ext)

    def get_time(self, x, technologies_indices):
        self.__check_variables_shape(x)
        return cvxpy.sum(cvxpy.multiply(x[:, technologies_indices - 1],
                                        self.inv_P[:, technologies_indices - 1]))

    def get_time_1(self, x):
        self.__check_variables_shape(x)
        return self.get_time(x, self.N1)

    def get_time_2(self, x):
        self.__check_variables_shape(x)
        return self.get_time(x, self.N2_ext)

    def total_productivity(self, x):
        self.__check_variables_shape(x)
        return self.A_ext @ x.T
