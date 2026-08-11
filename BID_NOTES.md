# Ready-to-paste proposal snippet

> To show rather than tell, I've built a working demo for this bid: a
> location scoring engine covering **Greater Melbourne and Greater Sydney**,
> running on free public data (Overture Maps — no API keys or licensing
> costs). Live demo: `<HOSTED-URL>`
>
> It pulls ~31,000 POIs per city, aggregates them onto an H3 hex grid,
> clusters every cell into location archetypes with **KMeans** (CBD core,
> high-street strip, suburban retail hub…), and trains a **RandomForest** on
> the bubble tea stores already operating there (208 in Melbourne, 288 in
> Sydney) — learning what makes a location viable, then scoring every cell
> out-of-fold (**ROC-AUC 0.88 / 0.87**). Suitability discounted by existing
> supply surfaces ranked, underserved whitespace — see the red stars and the
> CSV shortlists on the maps.
>
> The full engine would layer in ABS Census demographics (age, income,
> cultural mix), foot-traffic proxies and rents, expose a `score(address)`
> API, and — once you can share per-store revenue — upgrade from
> "viability" to revenue prediction. Any Australian city is a one-line
> config change.

**Attach:** `output/melbourne_milk_tea_map.png` and
`output/sydney_milk_tea_map.png`; replace `<HOSTED-URL>` with the live demo
link once published.
