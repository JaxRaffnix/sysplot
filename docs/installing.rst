Installing
=====================
    
sysplot is available on PyPI and can be installed using pip:

.. code-block:: bash

   pip install sysplot

Alternatively, if you have cloned the repository, you can install it in development mode:

.. code-block:: bash

   pip install -e .

Quick Start
--------------

After installing, you can import sysplot in your Python code and start using it to create various types of plots for control systems analysis. For example, to create a Bode plot:
.. code-block:: python

    import matplotlib.pyplot as plt
    import control as ctrl
    import sysplot as ssp

    # Generate frequency response
    omega = np.logspace(-1, 8, 2000)
    system = ctrl.tf([1, 100], [1, 10])
    mag, phase, _ = ctrl.frequency_response(system, omega)

    # Create a figure with automatic sizing
    fig, axes = plt.subplots(figsize=ssp.get_figsize(nrows=1, ncols=2))
    ssp.highlight_axes(fig)

    # Plot the Bode diagram
    ssp.plot_bode(mag, phase, omega, axes=axes)
    fig.suptitle("Bode Plot in dB")
    axes[0].set_xlabel("Frequency [rad/s]")
    axes[0].set_ylabel("Amplitude [dB]")
    axes[1].set_ylabel("Phase [deg]")

    # Save the figure
    ssp.save_current_figure(chapter=1, number=1, folder="figures")
    plt.show()

.. image:: auto_examples/images/sphx_glr_plot_example_004.png
   :alt: Bode Plot
