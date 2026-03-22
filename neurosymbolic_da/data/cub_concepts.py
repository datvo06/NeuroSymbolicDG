"""LLM-generated concept bank for CUB-200 bird classification.

Discriminative visual features for fine-grained bird species recognition,
designed to be encoded by CLIP text encoder as concept primitives.

Generated via prompt: "List the key discriminative visual features
for classifying 200 bird species. Focus on spatial, part-based features
that vary across species and are visually detectable."
"""

# 64 bird-part concepts organized by body region
# Each concept is a short visual description suitable for CLIP text encoding

CUB_CONCEPTS = [
    # Head (12 concepts)
    "a bird with a pointed bill",
    "a bird with a curved bill",
    "a bird with a thick conical bill",
    "a bird with a long thin bill",
    "a bird with a hooked bill",
    "a bird with a red head",
    "a bird with a crested head",
    "a bird with a striped crown",
    "a bird with large eyes",
    "a bird with an eye ring",
    "a bird with a face mask pattern",
    "a bird with a colorful throat patch",

    # Breast and belly (8 concepts)
    "a bird with a spotted breast",
    "a bird with a red breast",
    "a bird with a yellow belly",
    "a bird with a white belly",
    "a bird with streaked underparts",
    "a bird with a banded chest",
    "a bird with a plain breast",
    "a bird with an orange breast",

    # Wings (10 concepts)
    "a bird with long pointed wings",
    "a bird with rounded wings",
    "a bird with wing bars",
    "a bird with colorful wing patches",
    "a bird with black wing tips",
    "a bird with striped wing feathers",
    "a bird with iridescent wings",
    "a bird with broad wings",
    "a bird with white wing edges",
    "a bird with folded wings showing pattern",

    # Tail (8 concepts)
    "a bird with a long forked tail",
    "a bird with a short squared tail",
    "a bird with a fanned tail",
    "a bird with tail spots",
    "a bird with a graduated tail",
    "a bird with white outer tail feathers",
    "a bird with a deeply notched tail",
    "a bird with colorful tail feathers",

    # Back and upperparts (6 concepts)
    "a bird with a striped back",
    "a bird with a plain colored back",
    "a bird with a spotted back",
    "a bird with iridescent upperparts",
    "a bird with a contrasting rump patch",
    "a bird with scaled back feathers",

    # Legs and feet (4 concepts)
    "a bird with long legs",
    "a bird with webbed feet",
    "a bird with yellow legs",
    "a bird with perching feet on a branch",

    # Overall shape and size (8 concepts)
    "a small compact bird",
    "a large bird of prey",
    "a slender elongated bird",
    "a stocky round bird",
    "a bird with an upright posture",
    "a bird in a horizontal posture",
    "a bird with a long neck",
    "a bird with a short neck",

    # Color patterns (8 concepts)
    "a mostly black bird",
    "a mostly white bird",
    "a brightly colored bird",
    "a dull brown bird",
    "a bird with contrasting black and white plumage",
    "a bird with blue plumage",
    "a bird with green plumage",
    "a bird with mixed brown and gray plumage",
]

# Subset sizes for different experiments
CONCEPTS_K8 = CUB_CONCEPTS[:8]    # minimal
CONCEPTS_K16 = CUB_CONCEPTS[:16]  # head + breast
CONCEPTS_K32 = CUB_CONCEPTS[:32]  # head + breast + wings + tail
CONCEPTS_K64 = CUB_CONCEPTS       # full set
