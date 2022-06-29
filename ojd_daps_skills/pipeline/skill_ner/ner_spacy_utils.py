def edit_ents(text, orig_ents):
    """
    A function to fix the text and entity spans,
    will remove trailing whitespace/punctuation
    from the text and spans
    """

    editted = False
    # Don't include trailing whitespace from entity spans
    trim_chars = [
        " ",
        ".",
        ",",
        ";",
        ":",
        "\xa0",
    ]  # any trailing chars that match these are removed
    trimmed_ents = []
    for b, e, l in orig_ents:
        if text[b] in trim_chars:
            new_b = b + 1
            editted = True
        else:
            new_b = b

        if text[e - 1] in trim_chars:
            new_e = e - 1
            editted = True
        else:
            new_e = e
        trimmed_ents.append((new_b, new_e, l))
    return trimmed_ents, editted


def fix_formatting_entities(text, ents):
    """
    Clean the text and entity spans for cases
    where the entity ends but the next character is not a space
    e.g. "this is OK you need to fixMe please and hereToo please"
    ents = [(8, 10, "LABEL"), (15, 26, "LABEL"), (36,44,"LABEL")]

    Also:
    - if start or end of entity is a space then trim it
    """
    ent_additions = [0] * len(ents)
    insert_index_space = []
    for i, (b, e, l) in enumerate(ents):

        # If the char before the start of this span is not a space,
        # Then update from this ent onwards
        if text[b - 1] != " ":
            ent_additions[i:] = [ea + 1 for ea in ent_additions[i:]]
            insert_index_space.append(b)

        # If the next char after this span is not a space,
        # then update the start and endings of all entities after this
        if (e) < len(text):
            if text[e] != " ":
                ent_additions[(i + 1) :] = [ea + 1 for ea in ent_additions[(i + 1) :]]
                insert_index_space.append(e)

    # Fix entity spans
    new_ents = []
    for (b, e, l), add_n in zip(ents, ent_additions):
        new_ents.append((b + add_n, e + add_n, l))

    # Add spaces in the correct places
    b = 0
    new_texts = []
    for e in insert_index_space:
        new_texts.append(text[b:e])
        b = e
    new_texts.append(text[b:])
    new_text = " ".join(new_texts)

    editted = True
    trimmed_ents = new_ents
    while editted:
        trimmed_ents, editted = edit_ents(new_text, trimmed_ents)

    return new_text, trimmed_ents
