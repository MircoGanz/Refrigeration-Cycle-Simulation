import numpy as np
from CoolProp.CoolProp import PropsSI
import joblib


def lam_w_correlation(Tw):
    """
    Stainless Steel AISI304 Thermal Conductivity Correlation deduced from
    Incropera F. P., Dewitt D. P., Bergman, T. L., Lavine A.S., 2017. Incropera's Principles of
    Heat and Mass Transfer 8th edition. Table A.1, p. 915

    Tw in K
    """
    a = 3.61867135e-02
    b = - 2.84400160e-05
    c = 1.13037051e-08
    d = 6.10135027e+00

    if 100 < Tw < 600:
        status = 1
    else:
        status = 0

    return a * Tw + b * Tw ** 2 + c * Tw ** 3 + d, status  # thermal conductivity of the plates [W/m K]


def alpha_correlation(ANN, S, D_h, n, p, h, T, Tw, Tc, m, phase, fluid):
    # print('Alpha Correlation')
    status = []
    if fluid == 'R134a':
        # print(fluid)
        if phase == "two-phase":
            # print(phase)
            if ANN:
                """ ANN """

                # noname
                rho_l = PropsSI('D', 'T', T, 'Q', 0, fluid)
                rho_v = PropsSI('D', 'T', T, 'Q', 1, fluid)
                rho = (rho_l + rho_v) / 2
                mu = (PropsSI('V', 'T', T, 'Q', 0, fluid) + PropsSI('V', 'T', T, 'Q', 1, fluid)) / 2

                noname_h = (rho * 9.81 * (rho_l - rho_v) * D_h ** 3) / mu ** 2
                # if 33725336.52303479 < noname_h < 51879348.27777562:
                if 30000000 < noname_h < 52000000:
                    status.append(1)
                else:
                    status.append(0)

                # Jakob number
                cp = (PropsSI('C', 'T', T, 'Q', 0, fluid) + PropsSI('C', 'T', T, 'Q', 1, fluid)) / 2
                deltaT = abs(Tw - Tc)
                h_fg = PropsSI('H', 'P', p, 'Q', 1, fluid) - PropsSI('H', 'P', p, 'Q', 0, fluid)

                Ja_h = cp * deltaT / h_fg
                if 0.009 < Ja_h < 0.0229:  # 0.00927541957939113 < Ja_h < 0.02287892022460113:
                    status.append(1)
                else:
                    status.append(0)

                # Prandtl number
                Pr_h = (PropsSI('Prandtl', 'T', T, 'Q', 0, fluid) + PropsSI('Prandtl', 'T', T, 'Q', 1, fluid)) / 2
                if 2 < Pr_h < 2.3:  # 2.0881923931971045 < Pr_h < 2.24857841981196
                    status.append(1)
                else:
                    status.append(0)

                # Bond number
                sigma = PropsSI('I', 'P', p, 'H', h, fluid)

                Bo_h = (9.81 * (rho_l - rho_v) * D_h ** 2) / sigma
                if 19 < Bo_h < 27:  # 19.63970379272617 < Bo_h < 26.85656860261496
                    status.append(1)
                else:
                    status.append(0)

                # Reynolds number
                G = m / S / n  # mass flux [kg/m^2 s]

                Re_h = G * D_h / mu
                if 265 < Re_h < 895:  # 270.10830161892204 < Re_h < 887.6046075654937
                    status.append(1)
                else:
                    status.append(0)

                # Nusselt number
                def Nu_h(transformerName, weights_file, X):

                    # transform input data
                    X = X.reshape(1, -1)
                    transf = joblib.load(transformerName + '.save')
                    X_trans = transf.transform(X)

                    # load weights
                    weights0 = np.load(f'{weights_file}_0.npy')
                    weights1 = np.load(f'{weights_file}_1.npy')
                    weights2 = np.load(f'{weights_file}_2.npy')
                    weights3 = np.load(f'{weights_file}_3.npy')

                    # manual prediction
                    X_trans = np.hstack(
                        [X_trans, np.zeros(X_trans.shape[0]).reshape(X_trans.shape[0], 1)])  # add extra bias column
                    X_trans[:, -1] = 1  # add placeholder as 1
                    weights = np.vstack([weights0, weights1])  # add trained weights as extra row vector
                    prediction = np.dot(X_trans, weights)  # now take dot product, repeat pattern for next layer
                    prediction[prediction < 0] = 0
                    prediction = np.hstack([prediction, np.zeros(prediction.shape[0]).reshape(prediction.shape[0], 1)])
                    prediction[:, -1] = 1
                    weights = np.vstack([weights2, weights3])
                    return np.dot(prediction, weights)[0][0]

                lam_h = PropsSI('L', 'P', p, 'Q', 0, fluid)  # liquid thermal conductivity [W/m K]

                return lam_h * Nu_h('quantiletrans', 'weights_n15',
                                    np.array([noname_h, Ja_h, Pr_h, Bo_h, Re_h])) / D_h, status  # [W/m2 K]
            else:
                """
                R134a Condensation Heat Transfer Correlation by
                Zhang J., Kærn M. R., Ommen T., Elmegaard B., Haglind F., 2019. Condensation heat transfer and pressure
                drop characteristics of R134a, R1234ze(E), R245fa and R1233zd(E) in a plate heat exchanger.
    
                """
                G = m / S / n  # mass flux [kg/m^2 s]
                x = PropsSI('Q', 'P', p, 'H', h, fluid)  # vapor quality [-]
                if x < 0:
                    x = 0
                rho_l = PropsSI('D', 'P', p, 'Q', 0, fluid)  # liquid density [kg/m^3]
                rho_v = PropsSI('D', 'P', p, 'Q', 1, fluid)  # vapor density [kg/m^3]
                G_eq = G * (1 - x + x * ((rho_l / rho_v) ** 0.5))  # equivalent mass flux [kg/m^2 s]
                mu_l = PropsSI('V', 'P', p, 'Q', 0, fluid)  # liquid dynamic viscosity [Pa s]

                Re_eq = (G_eq * D_h) / mu_l  # equivalent Reynolds number [-]
                if 1207 < Re_eq < 4827:
                    status.append(1)
                else:
                    status.append(0)

                Pr_l = PropsSI('Prandtl', 'P', p, 'Q', 0, fluid)  # liquid Prandtl number [-]
                if 3.1 < Pr_l < 6.5:
                    status.append(1)
                else:
                    status.append(0)

                sigma = PropsSI('I', 'P', p, 'Q', x, fluid)  # surface tension [N/m]

                Bo = (9.81 * (rho_l - rho_v) * (D_h ** 2)) / sigma  # Bond number [-]
                if 11.7 < Bo < 28.3:
                    status.append(1)
                else:
                    status.append(0)

                # Nusselt number [-], eq. (36)
                Nu = 4.3375 * (Re_eq ** 0.5383) * (Pr_l ** (1 / 3)) * (Bo ** (-0.3872))

                lambda_l = PropsSI('L', 'P', p, 'Q', 0, fluid)  # liquid thermal conductivity [W/m K]

                return (Nu * lambda_l) / D_h, status  # thermal transfer coefficient [W/m^2 K]

        elif phase == "vapor":
            # print(phase)
            if ANN:
                """ ANN """

                # noname
                if T > 374.21:
                    # print('T out of range')
                    rho_l = PropsSI('D', 'P', p, 'Q', 0, fluid)
                    rho_v = PropsSI('D', 'P', p, 'Q', 1, fluid)
                    mu = (PropsSI('V', 'P', p, 'Q', 0, fluid) + PropsSI('V', 'P', p, 'Q', 1, fluid)) / 2
                    cp = (PropsSI('C', 'P', p, 'Q', 0, fluid) + PropsSI('C', 'P', p, 'Q', 1, fluid)) / 2
                    Pr_h = (PropsSI('Prandtl', 'P', p, 'Q', 0, fluid) + PropsSI('Prandtl', 'P', p, 'Q', 1, fluid)) / 2

                else:
                    rho_l = PropsSI('D', 'T', T, 'Q', 0, fluid)
                    rho_v = PropsSI('D', 'T', T, 'Q', 1, fluid)
                    mu = (PropsSI('V', 'T', T, 'Q', 0, fluid) + PropsSI('V', 'T', T, 'Q', 1, fluid)) / 2
                    cp = (PropsSI('C', 'T', T, 'Q', 0, fluid) + PropsSI('C', 'T', T, 'Q', 1, fluid)) / 2
                    Pr_h = (PropsSI('Prandtl', 'T', T, 'Q', 0, fluid) + PropsSI('Prandtl', 'T', T, 'Q', 1, fluid)) / 2

                rho = (rho_l + rho_v) / 2

                noname_h = (rho * 9.81 * (rho_l - rho_v) * D_h ** 3) / mu ** 2
                if 34000000 < noname_h < 52000000:  # 33725336.52303479 < noname_h < 51879348.27777562
                    status.append(1)
                else:
                    status.append(0)

                # Jakob number
                deltaT = abs(Tw - Tc)
                h_fg = PropsSI('H', 'P', p, 'Q', 1, fluid) - PropsSI('H', 'P', p, 'Q', 0, fluid)

                Ja_h = cp * deltaT / h_fg
                if 0.009 < Ja_h < 0.023:  # 0.00927541957939113 < Ja_h < 0.02287892022460113
                    status.append(1)
                else:
                    status.append(0)

                # Prandtl number
                if 2 < Pr_h < 2.3:  # 2.0881923931971045 < Pr_h < 2.24857841981196
                    status.append(1)
                else:
                    status.append(0)

                # Bond number
                sigma = PropsSI('I', 'P', p, 'Q', 1, fluid)

                Bo_h = (9.81 * (rho_l - rho_v) * D_h ** 2) / sigma
                if 19 < Bo_h < 27:  # 19.63970379272617 < Bo_h < 26.85656860261496
                    status.append(1)
                else:
                    status.append(0)

                # Reynolds number
                G = m / S / n  # mass flux [kg/m^2 s]

                Re_h = G * D_h / mu
                if 265 < Re_h < 895:  # 270.10830161892204 < Re_h < 887.6046075654937
                    status.append(1)
                else:
                    status.append(0)

                # Nusselt number
                def Nu_h(transformerName, weights_file, X):

                    # transform input data
                    X = X.reshape(1, -1)
                    transf = joblib.load(transformerName + '.save')
                    X_trans = transf.transform(X)

                    # load weights
                    weights0 = np.load(f'{weights_file}_0.npy')
                    weights1 = np.load(f'{weights_file}_1.npy')
                    weights2 = np.load(f'{weights_file}_2.npy')
                    weights3 = np.load(f'{weights_file}_3.npy')

                    # manual prediction
                    X_trans = np.hstack(
                        [X_trans, np.zeros(X_trans.shape[0]).reshape(X_trans.shape[0], 1)])  # add extra bias column
                    X_trans[:, -1] = 1  # add placeholder as 1
                    weights = np.vstack([weights0, weights1])  # add trained weights as extra row vector
                    prediction = np.dot(X_trans, weights)  # now take dot product, repeat pattern for next layer
                    prediction[prediction < 0] = 0
                    prediction = np.hstack([prediction, np.zeros(prediction.shape[0]).reshape(prediction.shape[0], 1)])
                    prediction[:, -1] = 1
                    weights = np.vstack([weights2, weights3])
                    return np.dot(prediction, weights)[0][0]

                lam_h = PropsSI('L', 'P', p, 'Q', 0, fluid)  # liquid thermal conductivity [W/m K]

                return lam_h * Nu_h('quantiletrans', 'weights_n15',
                                    np.array([noname_h, Ja_h, Pr_h, Bo_h, Re_h])) / D_h, status
            else:
                """
                R134a Condensation Heat Transfer Correlation by
                Zhang J., Kærn M. R., Ommen T., Elmegaard B., Haglind F., 2019. Condensation heat transfer and pressure
                drop characteristics of R134a, R1234ze(E), R245fa and R1233zd(E) in a plate heat exchanger.

                """
                G = m / S / n  # mass flux [kg/m^2 s]
                # print('G: ', G)
                x = 1  # vapor quality [-]
                rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)  # liquid density [kg/m^3]
                rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)  # vapor density [kg/m^3]
                G_eq = G * (1 - x + x * ((rho_l / rho_v) ** 0.5))  # equivalent mass flux [kg/m^2 s]
                mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)  # liquid dynamic viscosity [Pa s]

                Re_eq = (G_eq * D_h) / mu_l  # equivalent Reynolds number [-]
                if 1207 < Re_eq < 4827:
                    status.append(1)
                else:
                    status.append(0)

                Pr_l = PropsSI('Prandtl', 'P', p, 'Q', 0.0, fluid)  # liquid Prandtl number [-]
                if 3.1 < Pr_l < 6.5:
                    status.append(1)
                else:
                    status.append(0)

                sigma = PropsSI('I', 'P', p, 'Q', x, fluid)  # surface tension [N/m]

                Bo = (9.81 * (rho_l - rho_v) * (D_h ** 2)) / sigma  # Bond number [-]
                if 11.7 < Bo < 28.3:
                    status.append(1)
                else:
                    status.append(0)

                # Nusselt number [-], eq. (36)
                Nu = 4.3375 * (Re_eq ** 0.5383) * (Pr_l ** (1 / 3)) * (Bo ** (-0.3872))

                lambda_l = PropsSI('L', 'P', p, 'Q', 0.0, fluid)  # liquid thermal conductivity [W/m K]

                return (Nu * lambda_l) / D_h, status  # thermal transfer coefficient [W/m^2 K]

        else:  # R134a, liquid
            # print(phase)

            # """ ANN """
            #
            # # noname
            # rho_l = PropsSI('D', 'T', T, 'Q', 0, fluid)
            # rho_v = PropsSI('D', 'T', T, 'Q', 1, fluid)
            # rho = (rho_l + rho_v) / 2
            # mu = (PropsSI('V', 'T', T, 'Q', 0, fluid) + PropsSI('V', 'T', T, 'Q', 1, fluid)) / 2
            #
            # noname_h = (rho * 9.81 * (rho_l - rho_v) * D_h ** 3) / mu ** 2
            # if 33725336.52303479 < noname_h < 51879348.27777562:
            #     status.append(1)
            # else:
            #     status.append(0)
            #
            # # Jakob number
            # cp = (PropsSI('C', 'T', T, 'Q', 0, fluid) + PropsSI('C', 'T', T, 'Q', 1, fluid)) / 2
            # deltaT = abs(Tw - Tc)
            # h_fg = PropsSI('H', 'P', p, 'Q', 1, fluid) - PropsSI('H', 'P', p, 'Q', 0, fluid)
            #
            # Ja_h = cp * deltaT / h_fg
            # if 0.00927541957939113 < Ja_h < 0.02287892022460113:
            #     status.append(1)
            # else:
            #     status.append(0)
            #
            # # Prandtl number
            # Pr_h = (PropsSI('Prandtl', 'T', T, 'Q', 0, fluid) + PropsSI('Prandtl', 'T', T, 'Q', 1, fluid)) / 2
            # if 2.0881923931971045 < Pr_h < 2.24857841981196:
            #     status.append(1)
            # else:
            #     status.append(0)
            #
            # # Bond number
            # sigma = PropsSI('I', 'P', p, 'Q', 0, fluid)
            #
            # Bo_h = (9.81 * (rho_l - rho_v) * D_h ** 2) / sigma
            # if 19.63970379272617 < Bo_h < 26.85656860261496:
            #     status.append(1)
            # else:
            #     status.append(0)
            #
            # # Reynolds number
            # G = m / S / n  # mass flux [kg/m^2 s]
            #
            # Re_h = G * D_h / mu
            # if 270.10830161892204 < Re_h < 887.6046075654937:
            #     status.append(1)
            # else:
            #     status.append(0)
            #
            # # Nusselt number
            # def Nu_h(transformerName, weights_file, X):
            #
            #     # transform input data
            #     X = X.reshape(1, -1)
            #     transf = joblib.load(transformerName + '.save')
            #     X_trans = transf.transform(X)
            #
            #     # load weights
            #     weights0 = np.load(f'{weights_file}_0.npy')
            #     weights1 = np.load(f'{weights_file}_1.npy')
            #     weights2 = np.load(f'{weights_file}_2.npy')
            #     weights3 = np.load(f'{weights_file}_3.npy')
            #
            #     # manual prediction
            #     X_trans = np.hstack(
            #         [X_trans, np.zeros(X_trans.shape[0]).reshape(X_trans.shape[0], 1)])  # add extra bias column
            #     X_trans[:, -1] = 1  # add placeholder as 1
            #     weights = np.vstack([weights0, weights1])  # add trained weights as extra row vector
            #     prediction = np.dot(X_trans, weights)  # now take dot product, repeat pattern for next layer
            #     prediction[prediction < 0] = 0
            #     prediction = np.hstack([prediction, np.zeros(prediction.shape[0]).reshape(prediction.shape[0], 1)])
            #     prediction[:, -1] = 1
            #     weights = np.vstack([weights2, weights3])
            #     return np.dot(prediction, weights)[0][0]
            #
            # lam_h = PropsSI('L', 'P', p, 'H', h, fluid)  # thermal conductivity [W/m K]
            #
            # return lam_h * Nu_h('quantiletrans_3v6', 'weights_n19',
            #                     np.array([noname_h, Ja_h, Pr_h, Bo_h, Re_h])) / D_h, status

            # """
            # Nu correlation empirically fitted with SWEP-tool data of the IWT
            # """
            # G = m / S / n  # mass flux [kg/m^2 s]
            # mu = PropsSI('V', 'P|liquid', p, 'T', T, fluid)  # dynamic viscosity [Pa s == N s / m2 == kg / m s]
            # Re_l = G * D_h / mu  # Reynolds number [-]
            # Pr_l = PropsSI('Prandtl', 'P', p, 'T', T, fluid)
            # if 90 < Re_l < 440:  # 94.751 < Re_l < 434.038
            #     status.append(1)
            # else:
            #     status.append(0)
            # if 0.8 < Pr_l < 0.9:  # 0.8437 < Pr_l < 0.8832
            #     status.append(1)
            # else:
            #     status.append(0)
            #
            # def Nu_h(Re, Pr):
            #     c = [1.13204430e-01, 4.33424174e+06, -5.33813353e-04, 6.62234974e+06,
            #          2.05343192e-06, -2.70479785e+07, -4.30418670e-09, 2.69288008e+07,
            #          3.59931425e-12, -8.85677157e+06, -1.98511532e+06]
            #
            #     return c[0] * Re + c[1] * Pr + c[2] * Re ** 2 + c[3] * Pr ** 2 + c[4] * Re ** 3 + c[5] * Pr ** 3 + \
            #            c[6] * Re ** 4 + c[7] * Pr ** 4 + c[8] * Re ** 5 + c[9] * Pr ** 5 + c[10]
            #
            # lam_l = PropsSI('L', 'P|liquid', p, 'H', h, fluid)  # thermal conductivity [W/m K]
            #
            # return (lam_l * Nu_h(Re_l, Pr_l)) / D_h, status  # thermal transfer coefficient [W/m^2 K]

            return 315, 1

    else:  # Antifrogen N, liquid
        # print(fluid)
        # print('Phase: ', phase)

        """
        Nu correlation empirically fitted with SWEP-tool data
        """
        G = m / S / n  # mass flux [kg/m^2 s]
        mu = PropsSI('V', 'P', p, 'T', T, fluid)  # dynamic viscosity [Pa s == N s / m2 == kg / m s]
        Re_c = G * D_h / mu  # Reynolds number [-]
        Pr_c = PropsSI('Prandtl', 'P', p, 'T', T, fluid)
        if 70 < Re_c < 1270:  # 74.9 < Re_c < 1265
            status.append(1)
        else:
            status.append(0)
        if 14 < Pr_c < 29:  # 14.3 < Pr_c < 28.7
            status.append(1)
        else:
            status.append(0)

        def Nu_c(Re, Pr):
            c = [2.01810008e-01, -5.40274165e+06, -3.60995238e-04, 1.19739190e+07, 6.69900486e-07, 1.14083644e+05,
                 -5.95058376e-10, -2.97014526e+07, 1.89744660e-13, 2.62508961e+07, 7.29705499e+05]

            return c[0] * Re + c[1] * Pr + c[2] * Re ** 2 + c[3] * Pr ** 2 + c[4] * Re ** 3 + c[5] * Pr ** 3 + \
                   c[6] * Re ** 4 + c[7] * Pr ** 4 + c[8] * Re ** 5 + c[9] * Pr ** 5 + c[10]

        lam_c = PropsSI('L', 'P', p, 'H', h, fluid)  # thermal conductivity [W/m K]
        return (lam_c * Nu_c(Re_c, Pr_c)) / D_h, status  # thermal transfer coefficient [W/m^2 K]


def pressure_loss(S, D_h, L, n, p, h, m, w, phase, fluid):
    # print('Pressure Loss')
    status = []
    L_cell = L * w  # length of the cell

    if phase == 'two-phase':
        if fluid == 'R134a':
            # print(fluid)
            # print('Phase: two-phase')
            """ 
            R134a Pressure Drop Correlation by
            Zhang J., Kærn M. R., Ommen T., Elmegaard B., Haglind F., 2019. Condensation heat transfer and pressure 
            drop characteristics of R134a, R1234ze(E), R245fa and R1233zd(E) in a plate heat exchanger.
            """

            G = m / S / n  # mass flux [kg/m^2 s]
            x = PropsSI('Q', 'P', p, 'H', h, fluid)  # vapor quality [-]
            rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)  # liquid density [kg/m^3]
            rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)  # vapor density [kg/m^3]
            G_eq = G * (1 - x + x * ((rho_l / rho_v) ** (1 / 2)))  # equivalent mass flux [kg/m^2 s]
            mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)  # liquid dynamic viscosity [Pa s]

            Re_eq = (G_eq * D_h) / mu_l  # equivalent Reynolds number [-]
            if 1207 < Re_eq < 4827:
                status.append(1)
            else:
                status.append(0)

            rho_m = 1 / ((x / rho_v) + ((1 - x) / rho_l))  # average two-phase density [kg/m3], eq. (24)
            sigma = PropsSI('I', 'P', p, 'H', h, fluid)  # surface tension [N/m]
            We = ((G ** 2) * D_h) / (rho_m * sigma)  # Weber number [-], eq. (39)
            if 1.8 < We < 68.2:
                status.append(1)
            else:
                status.append(0)

            f = 0.0146 * (Re_eq ** 0.9814) * (We ** (-1.0064))  # friction factor [-], eq. (40)

            # frictional pressure drop, Yan et al. 1999, eq. (29)
            return (2 * f * (G ** 2) * (1 / rho_m) * L_cell) / D_h, status
            # return 0

    elif phase == 'vapor':
        # print(fluid)
        # print('Phase: ', phase)
        if fluid == 'R134a':
            """ 
            R134a Pressure Drop Correlation by
            Zhang J., Kærn M. R., Ommen T., Elmegaard B., Haglind F., 2019. Condensation heat transfer and pressure 
            drop characteristics of R134a, R1234ze(E), R245fa and R1233zd(E) in a plate heat exchanger.
            """

            G = m / S / n  # mass flux [kg/m^2 s]
            x = 1  # vapor quality [-]
            rho_l = PropsSI('D', 'P', p, 'Q', 0.0, fluid)  # liquid density [kg/m^3]
            rho_v = PropsSI('D', 'P', p, 'Q', 1.0, fluid)  # vapor density [kg/m^3]
            G_eq = G * (1 - x + x * ((rho_l / rho_v) ** (1 / 2)))  # equivalent mass flux [kg/m^2 s]
            mu_l = PropsSI('V', 'P', p, 'Q', 0.0, fluid)  # liquid dynamic viscosity [Pa s]

            Re_eq = (G_eq * D_h) / mu_l  # equivalent Reynolds number [-]
            if 1207 < Re_eq < 4827:
                status.append(1)
            else:
                status.append(0)

            rho_m = 1 / ((x / rho_v) + ((1 - x) / rho_l))  # average two-phase density [kg/m3], eq. (24)
            # sigma = PropsSI('I', 'P', p, 'H', h, fluid)  # surface tension [N/m]
            sigma = PropsSI('I', 'P', p, 'Q', x, fluid)  # surface tension [N/m]
            We = ((G ** 2) * D_h) / (rho_m * sigma)  # Weber number [-], eq. (39)
            if 1.8 < We < 68.2:
                status.append(1)
            else:
                status.append(0)

            f = 0.0146 * (Re_eq ** 0.9814) * (We ** (-1.0064))  # friction factor [-], eq. (40)

            # frictional pressure drop, Yan et al. 1999, eq. (29)
            return (2 * f * (G ** 2) * (1 / rho_m) * L_cell) / D_h, status
        else:
            # print(fluid)
            # print('Phase: ', phase)
            return 0, status
    else:
        # print(fluid)
        # print('Phase: ', phase)
        return 0, status
