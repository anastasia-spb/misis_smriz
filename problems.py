import cvxpy
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from parameters import Parameters


class GenericProblem:
    def __init__(self):
        self.params = Parameters()
        self.precision = 3

    def plan_constraints(self, x):
        return [(x >= 0), (self.params.total_productivity(x) >= self.params.plan.T)]

    @staticmethod
    def resources_sum(x):
        return cvxpy.sum(x).value

    def print_resources_sum(self, x):
        print("Общее количество ресурсов: {} тонн".format(np.round(GenericProblem.resources_sum(x), self.precision)))

    def print_time(self, x):
        print("Первый агрегат: {} часов.".format(np.round(self.params.get_time_1(x).value, self.precision)))
        print("Второй агрегат: {} часов.".format(np.round(self.params.get_time_2(x).value, self.precision)))

    def print_products(self, x):
        products = np.round(self.params.total_productivity(x).value, self.precision)
        print("Количество произведенного продукта: {}, {}, {}, {}, {}, {}".format(products[0], products[1], products[2],
                                                                                  products[3], products[4],
                                                                                  products[5]))

    def print_results(self, problem, x):
        if problem.status not in ["infeasible", "unbounded"]:
            self.print_time(x)
            self.print_resources_sum(x)
            self.print_products(x)
        else:
            print("Проблема не имеет решения")

    def solve_problem(self, problem, x):
        problem.solve()
        self.print_results(problem, x)


class SolveMinResourceProblem(GenericProblem):
    def __init__(self):
        super().__init__()
        self.x = cvxpy.Variable(shape=(1, self.params.N_ext), integer=False)
        self.problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(self.x)),
                                     constraints=self.plan_constraints(self.x))

    def solve(self):
        super().solve_problem(self.problem, self.x)
        return np.round(super().resources_sum(self.x), self.precision)


class MinimizeWorkingTimeProblem(GenericProblem):
    def __init__(self):
        super().__init__()
        self.x = cvxpy.Variable(shape=(1, self.params.N_ext), integer=False)
        self.problem = cvxpy.Problem(cvxpy.Minimize(self.params.get_time_1(self.x) +
                                                    cvxpy.abs(self.params.get_time_1(self.x) - self.params.get_time_2(
                                                        self.x))),
                                     constraints=self.plan_constraints(self.x))

    def solve(self):
        super().solve_problem(self.problem, self.x)


@dataclass
class Prices:
    extra_time_1: float = 1.0
    extra_time_2: float = 1.0
    resource_price: float = 1.0


class MinimizeTimeAndResources(GenericProblem):
    def __init__(self, minimum_resource, prices: Prices):
        super().__init__()
        self.x = cvxpy.Variable(shape=(1, self.params.N_ext), integer=False)
        self.delta = cvxpy.Parameter(nonneg=True)
        self.delta.value = 30
        self.min_resource = minimum_resource
        self.prices = prices

        constraints = self.plan_constraints(self.x) + [(cvxpy.sum(self.x) <= (self.min_resource + self.delta))]
        self.problem = cvxpy.Problem(cvxpy.Minimize(self.__expenses()), constraints=constraints)

    def __expenses(self):
        extra_time_1 = self.params.get_time_1(self.x) - self.params.T1
        extra_time_2 = self.params.get_time_2(self.x) - self.params.T2
        resources_cost = cvxpy.sum(self.x)
        return (self.prices.extra_time_1 * extra_time_1 + self.prices.extra_time_2 * extra_time_2 +
                self.prices.resource_price * resources_cost)

    def print_expenses(self):
        print("Общие расходы: {}".format(np.round(self.__expenses().value, self.precision)))

    @staticmethod
    def __grid_search(f, a: float, b: float):
        idx = 0
        ranges = np.linspace(a, b, 4)
        while abs(b - a) >= 0.01:
            _, idx = min((f(val), idx) for (idx, val) in enumerate(ranges))
            if idx < 2:
                b = ranges[2]
            else:
                a = ranges[1]
            ranges = np.linspace(a, b, 4)
        return (a + b) / 2

    def __warm_solve(self, new_delta):
        self.delta.value = new_delta
        self.problem.solve(warm_start=True)
        if self.problem.status not in ["infeasible", "unbounded"]:
            return np.round(self.__expenses().value, self.precision)
        else:
            return np.inf

    def solve(self):
        optimal_delta = self.__grid_search(lambda x: self.__warm_solve(x), 0.0, self.min_resource)
        self.__warm_solve(optimal_delta)
        self.print_results(self.problem, self.x)
        print("Оптимальное значение уступки: {}".format(np.round(self.delta.value, self.precision)))
        self.print_expenses()

    def calculate_min_price_per_product(self):
        # Используем нормализованную матрицу, так как необходимо разделить цену за ту часть
        # ресурса, что пошла на отходы, между всеми продуктами данной технологии
        resource_per_product = self.params.A_ext_norm @ self.x.T
        priced_t1 = self.prices.extra_time_1 * self.params.get_time_1_values(self.x)
        priced_t2 = self.prices.extra_time_2 * self.params.get_time_2_values(self.x)
        priced_time_1_per_product = self.params.A_ext_norm[:, self.params.N1 - 1] @ priced_t1.value.T
        priced_time_2_per_product = self.params.A_ext_norm[:, self.params.N2_ext - 1] @ priced_t2.value.T
        products = self.params.total_productivity(self.x).value
        priced_time_per_product = priced_time_1_per_product + priced_time_2_per_product
        priced_resource_per_product = self.prices.resource_price * resource_per_product.value
        total_price_per_product = (
                (priced_resource_per_product + priced_time_per_product) /
                products)

        total_price_per_product = total_price_per_product.reshape((self.params.M,))
        total_price_per_product_rounded = np.round(total_price_per_product, self.precision)
        print("Цена за единицу продукта, при которой прибыль будет равна нулю: \n", total_price_per_product_rounded)

        fig = plt.figure(1, figsize=(10, 8))
        ax = fig.add_subplot(111)
        products_idx = np.arange(1, self.params.M + 1, 1)
        ax.bar(products_idx, products.reshape(self.params.M, ), color='b', width=0.5)
        ax.bar(products_idx + 0.25, resource_per_product.value.reshape(self.params.M, ), color='m', width=0.5)
        ax.set_xticks(products_idx)
        ax.set_ylabel('Resource [tonnes]')
        plt.legend(['Resource per product', 'Including waste'])

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        ax2.bar(products_idx, priced_time_1_per_product.reshape(self.params.M, ), width=0.5, color='r')
        ax2.bar(products_idx, priced_time_2_per_product.reshape(self.params.M, ),
                bottom=priced_time_1_per_product.reshape(self.params.M, ), width=0.5, color='b')
        ax2.bar(products_idx, priced_resource_per_product.reshape(self.params.M, ),
                bottom=priced_time_1_per_product.reshape(self.params.M, ) + priced_time_2_per_product.reshape(
                    self.params.M, ), width=0.5, color='m')
        ax2.set_ylabel('Price')
        plt.legend(['Time 1', 'Time 2', 'Resource'])

        plt.show()
