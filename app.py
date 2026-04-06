import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ─────────────────────────────────────────────
# Globale instellingen
# ─────────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavusans"

MAX_STEPS = 1000

# ─────────────────────────────────────────────
# Gedeelde hulpfuncties (één definitie voor alle tabs)
# ─────────────────────────────────────────────
def f(x, y):
    """Differentiaalvergelijking: y' = y - x^2 + 1"""
    return y - x**2 + 1


def exact(x, y0):
    """Exacte oplossing behorend bij beginwaarde y0 op x0=0"""
    C = y0 - 1
    return (x + 1)**2 + C * np.exp(x)


def euler(x0, y0, h, n):
    """Euler methode; geeft arrays xs en ys terug"""
    xs = [x0]
    ys = [y0]
    for _ in range(n):
        y0 = y0 + h * f(x0, y0)
        x0 = x0 + h
        xs.append(x0)
        ys.append(y0)
    return np.array(xs), np.array(ys)


def exact_latex(y0):
    """Geeft LaTeX-string van de exacte oplossing terug (via sympy)"""
    x = sp.symbols('x')
    C = y0 - 1
    expr = (x + 1)**2 + C * sp.exp(x)
    return sp.latex(expr)


# ─────────────────────────────────────────────
# Tab 1 — Euler methode
# ─────────────────────────────────────────────
def euler_app():
    st.header("Euler methode")
    st.subheader("Gegeven beginwaardeprobleem")
    st.write("Gegeven de differentiaalvergelijking:")
    st.latex(r"y' = y - x^2 + 1")
    st.write("met beginvoorwaarde:")
    st.latex(r"y(x_0) = y_0")
    st.write(r"op het interval [$x_0$, $x_n$].")
    st.write(
        r"Varieer de stapgrootte of het aantal stappen om de Euler benadering "
        r"te zien t.o.v. de exacte oplossing."
    )

    y0 = st.number_input(
        "Beginwaarde $y_0$",
        min_value=-10.0, max_value=10.0, value=0.5, step=0.1,
        key="euler_y0"
    )

    st.write("De exacte oplossing bij de gegeven beginvoorwaarden:")
    st.latex(f"y = {exact_latex(y0)}")

    st.subheader("Interval instellingen")
    x_start = st.number_input("Start van interval $x_0$", value=0.0, key="euler_xstart")
    x_end   = st.number_input("Einde van interval $x_n$", value=2.0,  key="euler_xend")

    if x_end <= x_start:
        st.error("Einde van interval moet groter zijn dan start.")
        st.stop()

    interval_length = x_end - x_start

    mode = st.radio("Kies modus:", ["Stapgrootte $h$ kiezen", "Aantal stappen $n$ kiezen"])

    if mode == "Stapgrootte $h$ kiezen":
        h = st.slider("Stapgrootte $h$", 0.01, 1.0, 0.1, step=0.1)
        n = int(interval_length / h)
        st.write(f"Aantal stappen: {n}")
    else:
        n = st.slider("Aantal stappen $n$", 1, 200, 20)
        h = interval_length / n
        st.write(f"Stapgrootte: {h:.4f}")

    n = int(interval_length / h)
    if n > MAX_STEPS:
        st.warning("Te veel stappen → kies grotere $h$ of kleiner interval")
        st.stop()

    xs, ys = euler(x_start, y0, h, n)
    x_exact = np.linspace(x_start, x_end, 200)
    y_exact_vals = exact(x_exact, y0)

    st.subheader("Plot van de Euler benadering en exacte oplossing")
    fig, ax = plt.subplots()
    ax.plot(xs, ys, 'o-', label="Euler benadering")
    ax.plot(x_exact, y_exact_vals, '-', label="Exacte oplossing")
    ax.minorticks_on()

    show_grid = st.checkbox("Toon grid", value=True, key="euler_grid")
    if show_grid:
        ax.grid(True, which='major', linestyle='-',  alpha=0.6)
        ax.grid(True, which='minor', linestyle='--', alpha=0.3)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Foutenanalyse")
    error = np.abs(ys - exact(xs, y0))

    fig_err, ax_err = plt.subplots()
    ax_err.plot(xs, error, marker='o', label="Absolute fout")
    ax_err.set_title(r"Absolute fout $|y - y_{\mathrm{exact}}|$")
    ax_err.set_xlabel("$x$")
    ax_err.set_ylabel("Fout")
    ax_err.grid(True, linestyle='--', alpha=0.6)
    ax_err.legend()
    st.pyplot(fig_err)

    st.info("De fout neemt af bij kleinere stapgrootte h.")


# ─────────────────────────────────────────────
# Tab 2 — Foutenanalyse
# ─────────────────────────────────────────────
def fouten_app():
    st.header("Foutenanalyse (Euler methode)")
    st.markdown(
        r"De Euler methode is een numerieke benadering voor de exacte oplossing van een DV. "
        r"Bekijk voor verschillende stapgroottes hoe de fout zich ontwikkelt over het interval $[x_0, x_n]$."
    )

    x0    = st.number_input("Start x₀",        value=0.0, key="fouten_x0")
    y0    = st.number_input("Beginwaarde y₀",   value=0.5, key="fouten_y0")
    x_end = st.number_input("Eindpunt xₙ",      value=2.0, key="fouten_xend")

    if x_end <= x0:
        st.error("Eindpunt moet groter zijn dan startpunt")
        st.stop()

    st.markdown("### Kies stapgroottes")
    h_values = st.multiselect(
        "Vergelijk stapgroottes h",
        [0.5, 0.25, 0.1, 0.05],
        default=[0.5, 0.25, 0.1]
    )

    fig, ax = plt.subplots()

    for h in h_values:
        n = int((x_end - x0) / h)
        xs, ys = euler(x0, y0, h, n)
        error  = np.abs(exact(xs, y0) - ys)

        ax.plot(xs, error, marker='o', label=f"h = {h}")
        st.write(f"h = {h} → max absolute fout = {np.max(error):.4f}")

    ax.set_title("Fout |y - y_exact| voor verschillende h")
    ax.set_xlabel("x")
    ax.set_ylabel("Fout")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.info("Je ziet dat kleinere stapgroottes leiden tot kleinere fouten over het hele interval.")


# ─────────────────────────────────────────────
# Tab 3 — Taylor benadering
# ─────────────────────────────────────────────
def taylor_app():
    st.header("Euler methode vanuit Taylorbenadering")
    st.markdown(
        r"De Euler methode is een toepassing van een eerste orde Taylorbenadering van $y(x)$. "
        r"Varieer de stapgrootte en bekijk hoe de fout zich ontwikkelt."
    )
    st.latex(r"y(x + \Delta x) \approx y(x) + \Delta x \cdot y'(x)")

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("Startpunt x₀",    value=0.0, key="taylor_x0")
        y0 = st.number_input("Beginwaarde y₀",  value=0.5, key="taylor_y0")
    with col2:
        h = st.slider("Stap in x (Δx)", 0.05, 1.0, 0.3)

    # exacte oplossing met lokale y0 (Taylor tab heeft eigen y0)
    def exact_local(x):
        return exact(x, y0)

    mode = st.radio("Kies weergave:", ["Lokaal (1 stap)", "Globaal (meerdere stappen)"])

    if mode == "Lokaal (1 stap)":
        x1       = x0 + h
        y1_euler = y0 + h * f(x0, y0)
        y1_exact = exact_local(x1)

        x_vals = np.linspace(x0 - 0.5, x0 + 1.5, 200)

        fig, ax = plt.subplots()
        ax.plot(x_vals, exact_local(x_vals), label="Exacte oplossing")

        taylor_line = y0 + f(x0, y0) * (x_vals - x0)
        ax.plot(x_vals, taylor_line, linestyle="--", label="Raaklijn")

        ax.scatter([x0], [y0], color="black")
        ax.plot([x0, x1], [y0, y0], linestyle=":", color="gray")
        ax.text((x0 + x1) / 2, y0, "Δx", ha="center", va="bottom")

        ax.plot([x1, x1], [y0, y1_euler], color="orange")
        ax.scatter([x1], [y1_euler], color="orange", label="Euler")
        ax.scatter([x1], [y1_exact],  color="red",    label="Exact")

        ax.annotate(
            "",
            xy=(x1, y1_exact), xytext=(x1, y1_euler),
            arrowprops=dict(arrowstyle="<->", color="red", linewidth=2)
        )
        y_mid     = (y1_exact + y1_euler) / 2
        x_offset  = 0.015 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(x1 + x_offset, y_mid, "fout", color="red", va="bottom")

        ax.set_title("Één Euler stap")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.write(f"Absolute fout: {abs(y1_exact - y1_euler):.4f}")

    else:
        n_steps = st.slider("Aantal stappen", 1, 20, 5)
        h_step  = h / n_steps

        xs = [x0]
        ys = [y0]
        for _ in range(n_steps):
            ys.append(ys[-1] + h_step * f(xs[-1], ys[-1]))
            xs.append(xs[-1] + h_step)

        fig, ax = plt.subplots()
        x_vals = np.linspace(x0, xs[-1], 200)
        ax.plot(x_vals, exact_local(x_vals), label="Exacte oplossing")
        ax.plot(xs, ys, 'o-', label="Euler stappen")

        for i in range(n_steps):
            x_local = np.linspace(xs[i], xs[i] + h_step, 20)
            y_local = ys[i] + f(xs[i], ys[i]) * (x_local - xs[i])
            ax.plot(x_local, y_local, linestyle="--", alpha=0.5)

        ax.set_title("Euler als opeenvolgende Taylor stappen")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.info("Elke stap gebruikt een lokale lineaire benadering. Samen vormen deze de Euler benadering.")

    st.latex(r"y_{n+1} = y_n + \Delta x \cdot f(x_n, y_n)")


# ─────────────────────────────────────────────
# Tab 4 — Eigenvectoren
# ─────────────────────────────────────────────
def eigenvector_app():
    st.header("Eigenvectoren en Eigenwaarden")
    st.markdown("""
    Gegeven zijn de matrix $A$ en vector $v$.

    Varieer de getalswaarden en bekijk de eigenvectoren/eigenwaarden en de uitkomst in de plot.

    Een matrixvermenigvuldiging kan ook voor ieder punt in het vlak worden uitgevoerd
    en zichtbaar gemaakt worden in de plot. Het resultaat is een transformatie van het vlak
    die afhangt van de berekende eigenvectoren en eigenwaarden voor matrix $A$.
    """)

    st.markdown("### Invoer")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Matrix A**")
        mcol1, mcol2 = st.columns(2, gap="small")
        with mcol1:
            a11 = st.number_input("", value=2.0, key="a11", label_visibility="collapsed")
            a21 = st.number_input("", value=1.0, key="a21", label_visibility="collapsed")
        with mcol2:
            a12 = st.number_input("", value=1.0, key="a12", label_visibility="collapsed")
            a22 = st.number_input("", value=2.0, key="a22", label_visibility="collapsed")

        A = np.array([[a11, a12], [a21, a22]])
        st.latex(rf"A = \begin{{pmatrix}} {a11} & {a12} \\ {a21} & {a22} \end{{pmatrix}}")

    with col_right:
        st.markdown("**Vector v**")
        vcol1, vcol2 = st.columns(2, gap="small")
        with vcol1:
            st.markdown("vₓ")
            v1 = st.number_input("", value=1.0, key="v1", label_visibility="collapsed")
        with vcol2:
            st.markdown("vᵧ")
            v2 = st.number_input("", value=1.0, key="v2", label_visibility="collapsed")

        v_free = np.array([v1, v2])
        st.latex(rf"v = \begin{{pmatrix}} {v1} \\ {v2} \end{{pmatrix}}")

    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Complexe eigenwaarden afvangen
    if np.iscomplex(eigenvalues).any():
        st.warning(
            "Deze matrix heeft complexe eigenwaarden. "
            "De visualisatie werkt alleen voor reële eigenwaarden — pas de matrix aan."
        )
        st.stop()

    eigenvalues  = eigenvalues.real
    eigenvectors = eigenvectors.real

    st.subheader("Eigenparen")
    cols = st.columns(len(eigenvalues))
    for i, col in enumerate(cols):
        lam = eigenvalues[i]
        v   = eigenvectors[:, i]
        with col:
            st.latex(rf"\lambda_{{{i+1}}} = {lam:.3f}")
            st.latex(rf"v_{{{i+1}}} = \begin{{pmatrix}} {v[0]:.3f} \\ {v[1]:.3f} \end{{pmatrix}}")

    def plot_grid(ax, A, grid_range=5, n_lines=10):
        xs = np.linspace(-grid_range, grid_range, n_lines)
        for val in xs:
            # verticale lijn
            y      = np.linspace(-grid_range, grid_range, 100)
            x_vals = np.full_like(y, val)
            points = np.vstack((x_vals, y))
            transformed = A @ points
            ax.plot(x_vals, y, color="lightgray", linewidth=1)
            ax.plot(transformed[0], transformed[1], color="blue", alpha=0.4)
            # horizontale lijn
            x      = np.linspace(-grid_range, grid_range, 100)
            y_vals = np.full_like(x, val)
            points = np.vstack((x, y_vals))
            transformed = A @ points
            ax.plot(x, y_vals, color="lightgray", linewidth=1)
            ax.plot(transformed[0], transformed[1], color="blue", alpha=0.4)

    show_grid = st.checkbox("Toon transformatie van het vlak", True)
    fig, ax   = plt.subplots()

    if show_grid:
        plot_grid(ax, A)

    colors = ["tab:blue", "tab:orange"]
    for i in range(2):
        v  = eigenvectors[:, i]
        Av = A @ v
        ax.quiver(0, 0, v[0],  v[1],  angles='xy', scale_units='xy', scale=1,
                  color=colors[i], width=0.01,  label=f"v{i+1}")
        ax.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1,
                  color=colors[i], alpha=0.4, width=0.008)

    Av_free = A @ v_free
    ax.quiver(0, 0, v_free[0],  v_free[1],  angles='xy', scale_units='xy', scale=1,
              color="black", width=0.012, label="v")
    ax.quiver(0, 0, Av_free[0], Av_free[1], angles='xy', scale_units='xy', scale=1,
              color="gray",  alpha=0.6,  width=0.008, label="Av")

    ax.set_aspect('equal')

    all_vecs = []
    for i in range(2):
        all_vecs.extend([eigenvectors[:, i], A @ eigenvectors[:, i]])
    all_vecs.extend([v_free, Av_free])
    max_val = np.max(np.abs(all_vecs)) + 0.5

    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_title("Eigenvectoren en transformatie")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.75)
    st.pyplot(fig)

    st.info("Eigenvectoren behouden hun richting onder de transformatie A. Andere vectoren veranderen van richting.")
    st.info("De lichtgrijze lijnen tonen het originele rooster; de blauwe lijnen tonen de transformatie door A.")
    st.info("De eigenwaarden bepalen hoeveel het vlak in de richting van elke eigenvector wordt uitgerekt of ingekrompen.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    st.info("Deze app is bedoeld als ondersteuning bij het uitleggen van numerieke methoden en lineaire algebra.")

    tab1, tab2, tab3, tab4 = st.tabs(["Taylor", "Euler", "Fouten", "Eigenvectoren"])

    with tab1:
        taylor_app()
    with tab2:
        euler_app()
    with tab3:
        fouten_app()
    with tab4:
        eigenvector_app()


if __name__ == "__main__":
    main()
