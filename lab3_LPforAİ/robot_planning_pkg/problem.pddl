(define (problem robot_problem)
	(:domain robot_navigation)
	(:objects
		locationA locationB locationC - location
		robot1 - robot
	)

	(:init
		(at robot1 locationA)
		(connected locationA locationB)
		(connected locationB locationC)
	)

	(:goal
		(at robot1 locationC)
	)
)
