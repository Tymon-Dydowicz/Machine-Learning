(define (problem p1)
    (:domain world-of-blocks)
    (:objects a b c d e)
    (:init
        (clear c)
        (clear e)
        (on-top c b)
        (on-top b a)
        (on-top e d)
        (on-floor a)
        (on-floor d)
    )
    (:goal
        (on-top d b)
    )
)