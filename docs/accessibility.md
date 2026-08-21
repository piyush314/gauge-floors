# Accessibility statement

We want the experiments and tutorials to remain usable without color, fine
print, mouse input, or visual chart interpretation.

## Measures in this release

- Instructions use semantic headings, descriptive links, and text-first
  examples.
- Every empirical result is available as CSV or JSON with a data dictionary.
- Every generated chart has a concise text alternative and an adjacent long
  description linked to its source data.
- Charts use markers, line styles, hatching, outlines, or fill state in
  addition to color.
- SVG chart text remains text rather than converted paths.
- Tutorials are available as ordinary web pages and executable notebooks.
- Keyboard focus is visible in the Pages theme, and the navigation order
  follows the document structure.

## Known limitations

Mathematical accessibility varies across browsers, Markdown renderers, and
assistive technologies. Notebook code cells can also be verbose when read
linearly. The text-first tutorials, long descriptions, and machine-readable
tables are the primary alternatives; we do not yet claim formal WCAG
conformance.

The companion research papers are outside this repository. Requests for an
accessible paper format should be directed to the corresponding author.

## Validation

Release QA checks heading order, link targets, color-independent encodings,
chart text size, notebook execution, and correspondence between descriptions
and machine-readable data. Automated checks do not replace testing with actual
assistive technologies.

## Report a barrier

[Open an accessibility issue](https://github.com/piyush314/gauge-floors/issues/new?template=accessibility.yml)
with the affected page, the task you were trying to complete, and your preferred
format. For a private request, email `saopk@ornl.gov` with the subject
`gauge-floors accessibility`.
