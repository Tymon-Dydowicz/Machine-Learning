(define
    (domain world-of-blocks)
    (:requirements :adl)
    (:predicates
        (on-top ?x ?y)
        (on-floor ?x)
        (clear ?x)
        (picked-up ?x)
        (any-pickedup)
    )
    
    (:action pickup-from-block
        :parameters (?x ?z)
        :precondition
        (and
            (clear ?x)
            (on-top ?x ?z)
            (not (any-pickedup))
        )
        :effect
        (and
            (not (on-top ?x ?z))
            (picked-up ?x)
            (clear ?z)
            (any-pickedup)
        )
    )
    
    (:action pickup-from-floor
        :parameters (?x)
        :precondition
        (and
            (clear ?x)
            (on-floor ?x)
            (not (any-pickedup))
        )
        :effect
        (and
            (picked-up ?x)
            (any-pickedup)
        )
    )
    
    (:action putdown-to-floor
        :parameters (?x)
        :precondition
        (and
            (picked-up ?x)
            (any-pickedup)
        )
        :effect
        (and
            (not (picked-up ?x))
            (on-floor ?x)
            (clear ?x)
            (not (any-pickedup))
        )
    )
   
    (:action putdown-to-block
        :parameters (?y ?x)
        :precondition
        (and
            (picked-up?x)
            (clear ?y)
            (any-pickedup)
        )
        :effect
        (and
            (on-top ?x ?y)
            (not (clear ?y))
            (not (picked-up ?x))
            (not (any-pickedup))
        )
    )
    
)