
PrintBuildConfiguration()
ug_load_script("d3f_app/d3f_util.lua")

params = {
        numAnisoLvls            = util.GetParamNumber("-numAnisoLvls", 5),
        numRefs                         = util.GetParamNumber("-numRefs", 5),
        baseLvl                         = util.GetParamNumber("-baseLvl", 0),
        firstRedistProcs        = util.GetParamNumber("-firstRedistProcs", 64),
        redistProcs             = util.GetParamNumber("-redistProcs", 256),
        partitioner                     = util.GetParam("-partitioner", "dynamicBisection"), -- options are "dynamicBisection", "staticBisection", "parmetis"
        qualityRedistLevelOffset        = util.GetParamNumber("-redistLevelOffset", 5),
		qualityThreshold        = util.GetParamNumber("-qualityThreshold", 0.5),
		evaluationId = util.GetParam("-evaluationId",0),
		evaluationDir = util.GetParam("-communicationDir","."),
}

------------------------------------------
-- reading the parameters from file
--
-- can be used as p.porosity or the like
-- in the lua file from here on
--
-- this uses the ParameterUtil plugin
--
-- assembling the file name from the given information: directory and id
-- the id will be unique for a simulation run
---------------------------------------------
local p = util.Parameters:fromfile(params.evaluationDir.."/"..params.evaluationId.."_parameters.json")


function calculateMeasurementPoints(points,a,b)
	result = {}
	diff = (b-a)/(points+1)
	pos = a+diff
	for i = 0,points-1,1 do
		table.insert(result, {pos})
		pos = pos + diff
	end
	return result
end

function initial_lsf (x,y,t) return y-0.1 end
problem = 
{ 
	--------------------------------------
	-- The domain specific setup
	--------------------------------------
	domain = 
	{
		dim = 2,
		grid = "testbox-triangles-reoriented.ugx",
		numRefs = 5,
		numPreRefs = 0,
	},
	balancer = {
		qualityThreshold = params.qualityThreshold,
		partitioner = {
			type = params.partitioner,
			verbose = false,
			enableZCuts = false,
			clusteredSiblings = false
		},
		partitionPostProcessor = params.partitioner == "parmetis" and "clusterElementStacks" or "smoothPartitionBounds",
		hierarchy = {
			type = "standard",
			maxRedistProcs = params.redistProcs,
			minElemsPerProcPerLevel = 800,
			qualityRedistLevelOffset = params.qualityRedistLevelOffset,
			{
					upperLvl = 0,
					maxRedistProcs = params.firstRedistProcs
			},

			{
					upperLvl = 2,
					maxRedistProcs = 16
			},
	},
	},


	free_surface =
    {
		init_lsf = "initial_lsf",
		LSFOutflowSubsets = "LeftEdge,RightEdge",
		LSFDirichletSubsets = "BottomEdge,TopEdge",
		NumEikonalSteps = 16,
		InitNumEikonalSteps = 64,

		----------------------------------------------------
		-- Specify measurement of free surface position!
		--
		-- This uses the FSFileMeasurer C++ class available
		-- in new versions of the d3f plugin.
		----------------------------------------------------
		fs_height_points =
		{
			-- the file to write to.
			-- assembling the file name from the given information: directory and id
			csv_output = params.evaluationDir.."/"..params.evaluationId.."_measurement",

			-- the points to measure the free surface at
			data = calculateMeasurementPoints(10, 0, 1),

			-- dont compare the data to a taregt yet, we will do this in UGParameterEstimator
			compare = false,

			-- log output while measuring
			log = true
		},
		zero_init_nv = true,
		reinit_sdf_rate = 20,
		antideriv_src = true,
		debugOutput = true
	},

	flow = 
	{
		type = "haline",
		cmp = {"c", "p"},
		
		gravity = -9.81,           
		density = 					
		{	"linear", 				
			min = 997,				
			max = 998,				
		},	
		
		viscosity = 
		{	"const",
			min = 1e-3,	
			max = 1.5e-3,
			brine_max = 0.001		
		},
		
		diffusion		= 1.e-9,
		alphaL			= 0,
		alphaT			= 0,

		upwind 		= "partial", 
		boussinesq	= true,	

		porosity 		=  0.1,
		permeability 	=  1.0-12,

		{
			subset                  = {"Body"},

			---------------------------------------------------
			-- this is how you can use the parameters in code
			---------------------------------------------------
			porosity        =  p.porosity,
			permeability    =  p.permeability 
		},
		
		initial = 
		{
			{ cmp = "c", value = 0.0 },
			{ cmp = "p", value = 0.0 },		
		},
		
		boundary = 
		{
			natural = "noflux",
			
			{ cmp = "p", type = "level", bnd = "TopEdge", value = 0.0 },
		},
		
		source = 
		{
			---------------------------------------------------
			-- this is how you can use the parameters in code
			---------------------------------------------------
			{ point = {0.1, 0.05}, params = { p.inflow, 0} }
		}	
	},
	
	solver =
	{
		type = "newton",
		lineSearch = {
			type = "standard",
			maxSteps		= 4,
			lambdaStart		= 1,
			lambdaReduce	= 0.5,
			acceptBest 		= true,
			checkAll		= false
		},

		convCheck = {
			type		= "standard",
			iterations	= 10,
			absolute	= 1e-8,
			reduction	= 1e-6,
			verbose		= true
		},
		
		linSolver =
		{
			type = "bicgstab",
			precond = 
			{	
				type 		= "gmg",
				smoother 	= "ilu",
				cycle		= "V",
				preSmooth	= 3,
				postSmooth 	= 3,
				rap			= false,
				baseLevel	= 0,
				
			},
			convCheck = {
				type		= "standard",
				iterations	= 100,
				absolute	= 5e-9,
				reduction	= 1e-6,
				verbose		= true,
			}
		}
	},
	
	time = 
	{
		control	= "prescribed",
		start 	= 0.0,
		stop	= p.stoptime,
	  	dt      = 0.2,
		dtmin	= 0.2*0.01,
		dtmax	= 2,
		dtred	= 0.5,
		tol 	= 1e-2,
	},

	---------------------------------------
	-- might be overwritten, see below
	---------------------------------------
	output = 
	{
		freq	= 1,
		binary 	= true,
		{
			file = "vtk",
			type = "vtk",
	        data = {"c", "p", "q", "lsf", "nv", "nv_ext", "sdf", "lsf_cfl"},
		}

	}
} 

-------------------------------------------------
-- this parameter is set to 0 by default to 
-- avoid writing of unnecessary data
-- if you want output, enable it by setting it to 1 as a fixedparameter
-------------------------------------------------
if p.output ~= 1 then
	problem.output = nil
end

-- invoke the solution process
util.d3f.solve(problem);
