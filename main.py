from problems import SolveMinResourceProblem, MinimizeWorkingTimeProblem, MinimizeTimeAndResources, Prices

if __name__ == '__main__':
    print()
    print("Рассчитаем минимальное необходимое количество ресурса в тоннах для выполнения плана: ")
    min_resource = SolveMinResourceProblem()
    min_amount_of_resource = min_resource.solve()
    print()

    print("Рассчитаем оптимальное количество времени работы каждого агрегата: ")
    optimal_time = MinimizeWorkingTimeProblem()
    optimal_time.solve()
    print()

    print("Рассчитаем оптимальную величину уступки в тоннах: ")
    prices = Prices(3, 3, 1)
    optimal_delta = MinimizeTimeAndResources(min_amount_of_resource, prices)
    optimal_delta.solve()
    optimal_delta.calculate_min_price_per_product()
    print()
