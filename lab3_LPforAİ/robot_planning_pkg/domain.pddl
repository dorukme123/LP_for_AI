(define (domain robot_navigation)
	(:types
		robot location
	)
	
	(:predicates
		(at ?robot ?location)
		(connected ?from ?to)
	)

	(:action move
		:parameters (?robot ?from ?to)
		:precondition (and (at ?robot ?from) (connected ?from ?to))
		:effect (and (not (at ?robot ?from)) (at ?robot ?to))
	)
)
