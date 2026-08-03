:html_theme.sidebar_secondary.remove:

Semantic Types
==============

Semantic types name values by their domain meaning without changing their
runtime representation. Array types also document their expected shape and
data type.

Mimodium uses these types to connect tasks in a
`Dagreon Workflow <https://dagreon.readthedocs.io/en/latest/api.html#dagreon.Workflow>`_.
A task's ``__call__`` annotations identify its required inputs and produced
output, allowing Dagreon to execute only the tasks needed for a requested type.

.. mimodium-type-catalog::
