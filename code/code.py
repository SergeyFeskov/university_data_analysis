import pandas as pd
import numpy as np
import fractions as fr
import math as mt

def getDisRow():
    dis_series = pd.read_excel(r"D:\Sergey\3sem\ТВИМС\сбор данных\дискретные.xlsx",
                               names=['date', 'data'], usecols="A:B", header=1, squeeze=True, index_col=0)

    dis_series = dis_series[dis_series != '-']
    dis_series = dis_series.convert_dtypes()

    # print(f'Вариационный ряд:\n{sorted(dis_series)}')

    dis_arr = dis_series.to_numpy(dtype=np.int64, copy=True)

    var_row, freques = np.unique(ar=dis_arr, return_counts=True)
    volume = sum(freques)
    ref_freques = [fr.Fraction(numerator=freques[i], denominator=volume) for i in range(len(freques))]

    dis_row_dt = pd.DataFrame([var_row, freques, ref_freques],
                              index=['x', 'n', 'w'], columns=[('x'+str(i)) for i in range(1, len(var_row)+1)])

    return dis_row_dt

def getIntRow():
    int_series = pd.read_excel(r"D:\Sergey\3sem\ТВИМС\сбор данных\непрерывные.xlsx",
                               names=['date', 'data'], usecols="A:B", header=1, squeeze=True, index_col=0)

    #print(f'Вариационный ряд:\n{sorted(int_series)}')

    volume = len(int_series)
    int_num = mt.ceil(1+3.322*mt.log10(volume))
    #print(f'Количество интервалов: {int_num}')
    h = mt.ceil((int_series.max() - int_series.min()) / (int_num*10)) * 10
    #print(f'Шаг: {h}')
    freques, int_edges = np.histogram(int_series, bins=int_num, range=(int_series.min(), int_series.min()+h*int_num))
    int_edges = int_edges.astype(int)
    ref_freques = [fr.Fraction(numerator=freques[i], denominator=volume) for i in range(int_num)]
    #ints = [str(int(int_edges[i])) + '-' + str(int(int_edges[i + 1])) for i in range(int_num)]
    #pd.DataFrame([ints, freques, ref_freques], index=['a - a', 'n', 'w'], columns=[('a'+str(i)+'-a'+str(i+1)) for i in range(1, int_num+1)]).to_excel('continues.xlsx')
    return int_series.values, int_edges, freques, ref_freques

def getDistribFunc_dis(dis_row):
    func_arr = np.empty((dis_row.shape[1]+1, 3), dtype=fr.Fraction)
    func_arr[0][0] = 0
    func_arr[0][1] = dis_row.iat[0, 0]
    func_arr[0][2] = dis_row.iat[0, 0] - 50
    for i in range(1, dis_row.shape[1]):
        func_arr[i][0] = func_arr[i-1][0]+dis_row.iat[2, i-1]
        func_arr[i][1] = dis_row.iat[0, i]
        func_arr[i][2] = dis_row.iat[0, i-1]
    func_arr[dis_row.shape[1]][0] = 1
    func_arr[dis_row.shape[1]][1] = 950
    func_arr[dis_row.shape[1]][2] = 850
    return func_arr

def getDistribFunc_int(int_edges, ref_freques):
    h = int_edges[1]-int_edges[0]
    func_arr = np.empty((len(ref_freques)+3, 2), dtype=fr.Fraction)
    func_arr[0][0] = 0
    func_arr[0][1] = int_edges[0]-h//2
    func_arr[1][0] = 0
    func_arr[1][1] = int_edges[0]
    for i in range(2, len(ref_freques)+2):
        func_arr[i][0] = func_arr[i-1][0]+ref_freques[i-2]
        func_arr[i][1] = int_edges[i-1]
    func_arr[len(ref_freques)+2][0] = 1
    func_arr[len(ref_freques)+2][1] = int_edges.max() + h // 2
    return func_arr


# values, int_edges, int_freques, ref_freques = getIntRow()
# getDistribFunc_int(int_edges, ref_freques)
# print(int_edges)

def calcNumericChars_int():
    values, int_edges, int_freques, ref_freques = getIntRow()
    h = (int_edges[1]-int_edges[0])
    sample_mean = sum(values) / len(values)
    print(f'Выборочное среднее: {sample_mean}\n')

    moda_int = int_freques.argmax()
    print(f'Номер модального интервала: {moda_int+1}')
    moda = int_edges[moda_int]+h*\
           (int_freques[moda_int]-int_freques[moda_int-1])/\
           (2*int_freques[moda_int]-int_freques[moda_int-1]-int_freques[moda_int+1])
    print(f'Мода: {moda}')
    print(f'Частость моды: {ref_freques[moda_int]}\n')

    func_arr = getDistribFunc_int(int_edges, ref_freques)[1:-1]
    distrib_func_vals = func_arr[:, 0]
    med_val = sorted(distrib_func_vals, key=lambda w: abs(fr.Fraction(1, 2) - w))[0]
    print(f'Частость медианы: {med_val}')
    med_ind = np.where(distrib_func_vals == med_val)[0][0]
    print(f'Номер медианного интервала: {med_ind+1}')
    med = int_edges[med_ind]+h*(sum(ref_freques)/2-func_arr[med_ind, 0])/ref_freques[med_ind]
    print(f'Медиана: {float(med)}\n')

    disp = sum((values-sample_mean)**2)/len(values)
    print(f'Дисперсия: {disp}')
    stand_dev = mt.sqrt(disp)
    print(f'Среденеквадратичное отклонение: {stand_dev}\n')

    th_cent_mom = sum((values-sample_mean)**3)/len(values)
    print(f'Третий центральный момент: {th_cent_mom}')
    asymmetry = th_cent_mom / (stand_dev ** 3)
    print(f'Асимметрия: {asymmetry}\n')

    fo_cent_mom = sum((values-sample_mean)**4)/len(values)
    print(f'Четвёртый центральный момент: {fo_cent_mom}')
    excess = fo_cent_mom / (stand_dev ** 4) - 3
    print(f'Эксцесс: {excess}\n')

    var_coef = stand_dev / sample_mean
    print(f'Коэффицент вариации: {var_coef}\n')

    r = values.max()-values.min()
    print(f'Размах вариации: {r}\n')

    lin_dev = sum([mt.fabs(values[i]-sample_mean) for i in range(len(values))]) / len(values)
    print(f'Среднее линейное отклонение: {lin_dev}\n')

    values = np.sort(values)
    int_values = [values[np.array([1 if (int_edges[i] <= val < int_edges[i + 1]) else 0 for val in values]).nonzero()]
                  for i in range(len(int_edges) - 1)]
    ints = [str(int(int_edges[i])) + '-' + str(int(int_edges[i + 1])) for i in
                   range(len(int_freques))]
    group_means = [int_values[i].mean() for i in range(len(int_values))]
    group_disps = [sum((int_values[i]-group_means[i])**2)/len(int_values[i]) for i in range(len(int_values))]
    mean_group_disp = sum([group_disps[i]*int_freques[i] for i in range(len(int_values))])/sum(int_freques)
    intergroup_disp = sum([((group_means[i]-sample_mean)**2)*int_freques[i] for i in range(len(int_freques))])/sum(int_freques)
    # pd.DataFrame(np.asarray([group_means, group_disps]).transpose(), index=ints,
    #             columns=['Групповые средние', 'Групповые дисперсии']).to_excel('disp_sum_theory.xlsx')
    print(f'Средняя групповая дисперсия: {mean_group_disp}')
    print(f'Межгрупповая дисперсия: {intergroup_disp}')
    print(f'Эмпирический коэф. детерминации: {intergroup_disp/disp}')
    print(f'Sum of disps: {mean_group_disp+intergroup_disp}')


def calcNumericChars_dis(dis_row):
    sample_mean = sum([dis_row.iat[0,i]*dis_row.iat[2,i] for i in range(dis_row.shape[1])])
    print(f'Выборочное среднее: {float(sample_mean)}\n')

    moda_ind = dis_row.loc['n'].values.argmax()
    moda = dis_row.iat[0, moda_ind]
    print(f'Мода: {moda}')
    print(f'Частость моды: {dis_row.iat[2, moda_ind]}\n')

    func_arr = getDistribFunc_dis(dis_row)
    distrib_func_vals = func_arr[:, 0]
    med_val = sorted(distrib_func_vals, key= lambda w: abs(fr.Fraction(1, 2)-w))[0]
    print(f'Частость медианы: {med_val}')
    med_ind = np.where(distrib_func_vals == med_val)[0][0]
    med = (dis_row.iat[0, med_ind] + dis_row.iat[0, med_ind-1]) // 2
    print(f'Медиана: {med}\n')

    sec_st_mom = sum([(dis_row.iat[0, i]**2)*dis_row.iat[2, i] for i in range(dis_row.shape[1])])
    print(f'Второй начальный момент: {float(sec_st_mom)}')
    disp = sec_st_mom - sample_mean**2
    print(f'Дисперсия: {float(disp)}')
    stand_dev = mt.sqrt(disp)
    print(f'Среденеквадратичное отклонение: {stand_dev}\n')

    th_cent_mom = sum([((dis_row.iat[0, i]-sample_mean)**3)*dis_row.iat[2, i] for i in range(dis_row.shape[1])])
    print(f'Третий центральный момент: {float(th_cent_mom)}')
    asymmetry = th_cent_mom/(stand_dev**3)
    print(f'Асимметрия: {asymmetry}\n')

    fo_cent_mom = sum([((dis_row.iat[0, i]-sample_mean)**4)*dis_row.iat[2, i] for i in range(dis_row.shape[1])])
    print(f'Четвёртый центральный момент: {float(fo_cent_mom)}')
    excess = fo_cent_mom / (stand_dev ** 4) - 3
    print(f'Эксцесс: {excess}\n')

    var_coef = stand_dev/sample_mean
    print(f'Коэффицент вариации: {var_coef}\n')

    print(f'Отношение оценок дисперсии и мат.ожидания: {float(disp/sample_mean)}\n')
    print(f'|med - mean|: {mt.fabs(float(med-sample_mean))}')
    print(f'3|moda - mean|: {3*mt.fabs(float(moda - sample_mean))}\n')

    r = max(dis_row.loc['x'].values) - min(dis_row.loc['x'].values)
    print(f'Размах вариации: {r}\n')

    lin_dev = sum([mt.fabs(dis_row.iat[0, i]-sample_mean)*dis_row.iat[2, i] for i in range(dis_row.shape[1])])
    print(f'Среднее линейное отклонение: {lin_dev}\n')

def F(x, a, b):
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        return (x-a)/(b-a)

def getPlainDistrib():
    values, int_edges, int_freques, ref_freques = getIntRow()
    a = values.min()
    b = values.max()
    probs = [F(int_edges[i+1], a, b)-F(int_edges[i], a, b) for i in range(len(int_edges)-1)]
    return probs, a, b

def plainDistribToExcel():
    _, int_edges, int_freques, ref_freques = getIntRow()
    probs, _, _ = getPlainDistrib()
    df = pd.DataFrame(np.empty((len(probs), 4)),
                      columns=['a-a', 'n', 'w', 'p'],
                      index=[f'a{i}-a{i+1}' for i in range(len(probs))])
    df['a-a'] = [f'{int_edges[i]:.1f}-{int_edges[i+1]:.1f}' for i in range(len(probs))]
    df['n'] = int_freques
    df['w'] = ref_freques
    df['p'] = [f'{probs[i]:.2e}' for i in range(len(probs))]
    df.to_excel('plain.xlsx')

def getPyasonDistrib():
    dis_row = getDisRow()
    sample_mean = sum([dis_row.iat[0, i] * dis_row.iat[2, i] for i in range(dis_row.shape[1])])
    frequency = sample_mean / 70
    p = np.empty(18)
    x = np.array(range(0, 900, 50))
    for i in range(18):
        p[i] = (mt.pow(frequency, i) * mt.exp(-frequency)) / mt.factorial(i)
    return [x, p]

def getPyasonDistribFunc():
    cords = getPyasonDistrib()
    func_arr = np.empty((len(cords[0]) + 1, 3), dtype=fr.Fraction)
    func_arr[0][0] = 0
    func_arr[0][1] = cords[0][0]
    func_arr[0][2] = cords[0][0] - 50
    for i in range(1, len(cords[0])):
        func_arr[i][0] = func_arr[i - 1][0] + cords[1][i-1]
        func_arr[i][1] = cords[0][i]
        func_arr[i][2] = cords[0][i-1]
    func_arr[len(cords[0])][0] = 1
    func_arr[len(cords[0])][1] = 950
    func_arr[len(cords[0])][2] = 850
    return func_arr

# arr = getPyasonDistribFunc()
# print(arr)
# print(arr.shape)

def pyasonDistribToExcel():
    cords = getPyasonDistrib()
    #[cords[0], cords[0] // 50, [f'{cords[1][i]:.2e}' for i in range(18)]]
    df = pd.DataFrame(np.empty((18, 3)),
                      columns=['x', 'k', 'p'],
                      index=[f'x{i}(k{i})' for i in range(1, 19)])
    df['x'] = cords[0]
    df['k'] = cords[0] // 50
    df['p'] = [f'{cords[1][i]:.2e}' for i in range(18)]
    df.to_excel('pyason50.xlsx')

def getPirsonCrit_int():
    _, _, freques, _ = getIntRow()
    probs, _, _ = getPlainDistrib()
    n = sum(freques)
    crit = sum([mt.pow(freques[i] - n * probs[i], 2) / (n * probs[i]) for i in range(len(freques))])
    return crit

def getRomanovCrit_int():
    pirs_crit = getPirsonCrit_int()
    k = 5
    r = mt.fabs(pirs_crit-k) / (mt.sqrt(2*k))
    return r

def getJastremCrit_int():
    pirs_crit = getPirsonCrit_int()
    k = 5
    n = 8
    j = mt.fabs(pirs_crit-k) / (mt.sqrt(2*n+2.4))
    return j

# print(f'Критерий Пирсона: {getPirsonCrit_int()}')
# print(f'Критерий Романовского: {getRomanovCrit_int()}')
# print(f'Критерий Ястремского: {getJastremCrit_int()}')

def getPirsonCrit(th_distrib, stat_distrib):
    n = sum(stat_distrib.loc['n'].values)
    th_vals = th_distrib[0]
    th_probs = th_distrib[1]
    ints = [[0, 50], [100], [150], [200], [250, 300], [i for i in range(350, 500, 50)], [i for i in range(500, 900, 50)]]
    freques = [23, 25, 21, 7, 9, 5, 7]
    probs = [sum([th_probs[th_vals == val][0] for val in ints[i]]) for i in range(len(ints))]
    df = pd.DataFrame([['0-50', '100', '150', '200', '250-300', '350-450', '500-850'],
                       freques, [f'{probs[i]:.2e}' for i in range(len(probs))]],
                      index=['m', 'n', 'p'], columns=[f'm{i+1}' for i in range(len(ints))])
    df.to_excel('pirson.xlsx')
    crit = sum([mt.pow(freques[i]-n*probs[i], 2) / (n*probs[i]) for i in range(len(ints))])
    return crit

def getRomanovCrit():
    pirs_crit = getPirsonCrit(getPyasonDistrib(), getDisRow())
    k = 5
    r = mt.fabs(pirs_crit-k) / (mt.sqrt(2*k))
    return r

def getJastremCrit():
    pirs_crit = getPirsonCrit(getPyasonDistrib(), getDisRow())
    k = 5
    n = 7
    j = mt.fabs(pirs_crit-k) / (mt.sqrt(2*n+2.4))
    return j

# print(f'Критерий Пирсона: {getPirsonCrit(getPyasonDistrib(), getDisRow())}')
# print(f'Критерий Романовского: {getRomanovCrit()}')
# print(f'Критерий Ястремского: {getJastremCrit()}')



