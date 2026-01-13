# Architecture Diagram Generation Prompt

Use this prompt with an image generation model (e.g., nano banana) to regenerate the visual architecture overview when the system changes.

## Prompt

```
Steampunk fantasy software architecture diagram, dark industrial aesthetic with brass gears and pipes framing the edges.

Top section labeled "EVOLUTION LOOP" in ornate dashed gold border containing:
- "Initializer" (purple badge, clockwork alchemist character)
- "Population Pool" (wide central panel with bubbling alchemical vat, green magical vapors rising)
- Row of 4 agents below: "Mutator" (chaotic inventor), "Crossover" (fusion artist), "Evaluator" (mechanical judge), "Adversary" (sly rogue in hood)
- Flow arrows connecting them in a loop

Right side outside the loop:
- "Champion" in golden circular medallion frame with crowned knight character, arrow labeled "validated" pointing to it from Adversary

Bottom left section "DIAGNOSTIC AGENTS":
- "Debugger" (detective with magnifying glass, smoke/steam)
- "Plateau Breaker" (muscular steam-powered character)
- "Meta-Strategist" (wise sage with crystal ball, purple glow)

Bottom right section "SUPPORT SYSTEMS":
- "Memory System" (librarian character)
- "Trust Validation" (armored knight guards)
- Purple glowing "context injection" pipe/arrow connecting Memory System up to Mutator

Dark brown/bronze color palette, magical green and purple accents, detailed character art in each panel, readable text labels, 16:9 aspect ratio
```

## Character Metaphors

| Component | Visual Metaphor |
|-----------|-----------------|
| Initializer | Clockwork alchemist |
| Mutator | Chaotic inventor |
| Crossover | Fusion artist |
| Evaluator | Mechanical judge |
| Adversary | Sly rogue |
| Champion | Crowned knight |
| Debugger | Detective |
| Plateau Breaker | Steam-powered strongman |
| Meta-Strategist | Wise sage |
| Memory System | Librarian |
| Trust Validation | Armored guards |

## When to Regenerate

Regenerate the image when:
- Adding or removing agents
- Changing the data flow between components
- Adding new sections (e.g., new diagnostic agents)

Edit the relevant section of the prompt to match the new architecture.
