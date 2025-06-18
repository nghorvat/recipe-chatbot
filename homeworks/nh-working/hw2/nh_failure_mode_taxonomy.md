# Failure Mode Taxonomy

This document outlines the failure modes observed or anticipated for the Recipe Chatbot. Each failure mode includes a title, a concise definition, and illustrative examples.

## Failure Mode 1: Complete Recipe Generation Failure

*   **Definition**: Bot fails to provide any recipe at all, instead asking clarifying questions or providing only suggestions.
*   **Illustrative Examples**:
    1.  *User Query*: "Quick protein-packed breakfast ideas for busy mornings?"
        *Bot Response*: Provides a list of recipe options with descriptions, but no recipe generated.
    2.  *User Query*: "I need a healthy Mexican snack recipe that's low in sodium"
        *Bot Response*: Just asks a few follow-up questions without providing a recipe to start with.

## Failure Mode 2: Ingredient Sequencing Issues

*   **Definition**: Ingredients listed in alphabetical or random order rather than order of use, making recipes hard to follow while cooking.
*   **Illustrative Examples**:
    1.  *User Query*: "What Mexican breakfast works for someone avoiding carbs?"
        *Bot Response*: Provides a Mexican breakfast scramble recipe with the following ingredient list:
```
### Ingredients (2 servings)

- 4 large eggs
- 100 g (3.5 oz) Mexican chorizo (or substitute with cooked breakfast sausage or bacon)
- Â½ small onion, diced
- 1 small bell pepper (red or green), diced
- 1 small jalapeÃ±o, seeded and diced (optional, for heat)
- Â¼ cup shredded cheddar or Mexican blend cheese
- 1 small avocado, diced
- 2 tbsp fresh cilantro, chopped
- 2 tbsp salsa (choose your favorite low-carb salsa)
- 1 tbsp olive oil or butter
- Salt and pepper, to taste
```
...but the instructions uses the ingredients in an order different from above (olive oil / butter, then chorizo, then diced onion, bell pepper, jalapeno, etc.)
```
### Instructions

1. Heat olive oil or butter in a skillet over medium heat.
2. Add chorizo and cook until browned and cooked through, breaking it up with a spoon (about 5 minutes).
3. Add diced onion, bell pepper, and jalapeÃ±o. SautÃ© until softened, about 4-5 minutes.
...
``` 

## Failure Mode 3: Time Expectation Mismatches

*   **Definition**: Recipe timing doesn't match user's specified speed requirements, or bot redundantly asks about timing when already specified.
*   **Illustrative Examples**:
    1.  *User Query*: "Need urgent gluten-free Indian dinner recipe"
        *Bot Response*: Provides an Indian curry recipe that requires 20-25 minutes of pressure cooking chickpeas, 15 minutes of rice cooking, and 30+ minutes for cooking chana masala.
    2.  *User Query*: "What Mexican dinner can I make quickly while dieting?"
        *Bot Response*: Asks "How quick do you want it? Under 30 minutes, 15 minutes, or even faster?" as a follow-up question, when the timing of "quickly" is defined in the question.

## Failure Mode 4: Missing Nutritional Estimates

*   **Definition**: Lacks required protein, calorie, or sodium estimates when user specifically requests high/low content recipes.
*   **Illustrative Examples**:
    1.  *User Query*: "What's a low-calorie Indian breakfast for busy mornings?"
        *Bot Response*: Provides a recipe for Besan Chilla, the number of servings, but provides no estimated calories per serving.
    2.  *User Query*: "What's a speedy Italian lunch without too much sodium?"
        *Bot Response*: Provides a recipe for Italian Chickpea Salad with Grilled Chicken and Fresh Herbs, the number of servings, but provides no estimated sodium per serving.

## Failure Mode 5: Undefined Cooking Terminology

*   **Definition**: Uses technical cooking terms (like "al dente," "double boiler," "grind") without explanation for non-expert users.
*   **Illustrative Examples**:
    1.  *User Query*: "What Italian dinner can I make quickly without meat?"
        *Bot Response*: Provides a pasta recipe instructing to cook pasta until al dente, with no explanation of what this means.
    2.  *User Query*: "Need a fancy gluten-free Italian dessert for dinner party"
        *Bot Response*: Provides a recipe for tiramisu instructing to place a mixing bowl over a pot of simmering water (double boiler), with no explanation of how to do this safely and properly.


## Failure Mode 6: Missing Equipment Disclosure

*   **Definition**: Fails to mention required special tools, pans, or appliances upfront in ingredients list, discovered only mid-recipe.
*   **Illustrative Examples**:
    1.  *User Query*: "What's a low-calorie Indian breakfast for busy mornings?"
        *Bot Response*: Provides a recipe for Besan Chilla instructing to use a blender to make chutney, with no upfront alert that you need a blender to complete the recipe.
    2.  *User Query*: "Can you give me a complex Italian dessert recipe without gluten?"
        *Bot Response*: Provides a recipe for tiramisu instructing to transfer batter to a piping bag for piping, with no upfront alert that you need a piping bag to complete the recipe.

## Failure Mode 7: Imprecise Measurements

*   **Definition**: Uses vague measurements like "pinch" or "splash" instead of specific quantities, or uses ranges instead of single measurements.
*   **Illustrative Examples**:
    1.  *User Query*: "Can you give me a keto-friendly Mexican breakfast recipe?"
        *Bot Response*: Provides a recipe for a breakfast scramble instructing to "whisk the eggs with a pinch of salt and pepper".
    2.  *User Query*: "Can you give me a complex Italian dessert recipe without gluten?"
        *Bot Response*: Provides a recipe with one ingredient being "2-3 tbsp coffee liqueur". A specific number should be provided instead of a range.

## Failure Mode 8: Poor Recipe Workflow Organization

*   **Definition**: Complex recipes without clear sequencing or guidance on managing parallel cooking tasks.
*   **Illustrative Examples**:
    1.  *User Query*: "What's a satisfying vegan Indian lunch dish?"
        *Bot Response*: Provides a recipe for Vegan Chana Masala with Basmati Rice and provides four sets of instructions:
```
### Instructions

1. **Prepare chickpeas:**
   - If using dried chickpeas, soak overnight in plenty of water. Drain, then boil or pressure cook until tender (about 40-50 minutes boiling or 15-20 minutes in a pressure cooker). If using canned, rinse and drain well.

2. **Cook Basmati rice:**
   - Rinse rice under cold water until water runs clear.
   - In a pot, bring 2 cups water and a pinch of salt to a boil.
   - Add rice, reduce heat to low, cover, and cook for 15 minutes or until water is absorbed and rice is tender.
   - Fluff with a fork and set aside.

3. **Make the chana masala:**
   - Heat oil in a pan over medium heat. Add cumin seeds; let them sizzle for 30 seconds.
   - Add chopped onions and sautÃ© until golden brown (7-8 minutes).
   - Stir in garlic, ginger, and green chili; sautÃ© for 1-2 minutes until fragrant.
   - Add chopped tomatoes and cook until they soften and oil starts separating from the mixture (about 5-7 minutes).
   - Add coriander powder, turmeric, red chili powder, and salt; stir well.
   - Add the cooked chickpeas and mix to coat well with the masala.
   - Pour in water and bring to a simmer. Cook on medium-low 10-15 minutes to let the flavors meld.
   - Sprinkle garam masala and amchur powder or lemon juice; stir and cook another 2 minutes.
   - Adjust salt and spice as needed.

4. **Serve:**
   - Plate the basmati rice, ladle the warm chana masala over or beside it.
   - Garnish with fresh cilantro.
   - A simple cucumber salad or sautÃ©ed greens can be a refreshing side.
```
It's unclear if I should be completing these steps in sequence or in parallel. Instructions should specifically direct what to do in a sensible manner.

## Failure Mode 9: Content-Request Misalignment

*   **Definition**: Recipe content doesn't match requested meal type, dietary restrictions, or difficulty level.
*   **Illustrative Examples**:
    1.  *User Query*: "What's a traditional Italian breakfast I can make plant-based?"
        *Bot Response*: Provides a recipe for Vegan Italian Cornetto, a croissant-style pastry, which is a snack, not a breakfast.
    2.  *User Query*: "What Mexican snacks can I make that are heart-healthy?"
        *Bot Response*: Doesn't mention low-sodium at all in its reponse, a key factor in heart healthy diets.
    3.  *User Query*: "Looking for easy keto American dinner recipes"
        *Bot Response*: Provides a complex recipe involving a lot of steps, time and ingredients:
```
## Keto Bacon Cheeseburger Casserole

A savory, cheesy casserole packed with ground beef, crispy bacon, and melted cheese, this dish tastes just like a cheeseburger without the bun. Perfect for a satisfying keto dinner thatâ€™s quick to prepare.

### Ingredients (serves 4)

- 1 lb (450g) ground beef (80/20 fat ratio preferred)
- 6 slices bacon, chopped
- 1 cup shredded cheddar cheese
- Â½ cup shredded mozzarella cheese
- 1 small onion, finely chopped
- 2 cloves garlic, minced
- 3 large eggs
- Â½ cup heavy cream
- 1 tablespoon mustard (Dijon or yellow)
- 1 teaspoon Worcestershire sauce
- Salt and pepper to taste
- 2 cups fresh spinach (optional, for added greens)
- 2 green onions, sliced (for garnish)

### Instructions

1. Preheat oven to 350Â°F (175Â°C).
2. In a large skillet over medium heat, cook bacon until crispy. Remove and set aside on paper towels.
3. In the same skillet, add chopped onion and garlic; cook 2-3 minutes until softened.
4. Add ground beef to skillet and brown, breaking it apart as it cooks. Drain excess fat if necessary.
5. Stir in mustard, Worcestershire sauce, salt, and pepper to the beef mixture. Remove from heat.
6. In a bowl, whisk together eggs and heavy cream until combined.
7. Stir the cooked bacon, shredded cheddar, mozzarella, and spinach (if using) into the beef mixture.
8. Transfer mixture into a greased 9x9-inch baking dish. Pour egg and cream mixture evenly over the top.
9. Bake for 20-25 minutes or until the casserole is set and golden on top.
10. Let cool for a few minutes, garnish with sliced green onions, and serve.
```