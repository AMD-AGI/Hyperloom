{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{# Recurse into sub-modules, but skip test suites and example scripts so the
   API reference stays focused on the importable library surface. #}
{% set documented_modules = [] %}
{% for item in modules %}
{% set leaf = item.split('.')[-1] %}
{% if not leaf.startswith('test_') and leaf not in ['tests', 'examples', 'conftest'] %}
{{ documented_modules.append(item) or '' }}
{% endif %}
{% endfor %}
{% if documented_modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in documented_modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
